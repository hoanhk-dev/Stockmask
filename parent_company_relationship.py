import dspy
import json
from pydantic import BaseModel, field_serializer
from typing import Literal

import dspy
from pydantic import BaseModel, field_serializer
from typing import Literal, List

lm = dspy.LM(
    model="openai/gemini-2.5-flash",
    api_base="http://localhost:8317/v1",
    api_key="proxypal-local",
)
dspy.configure(lm=lm)

# -----------------------------
# Aspect structure
# -----------------------------
class ScoringAspect(BaseModel):
    aspect: str
    score: Literal[0, 1, 2, 3, 4, 5]   # allow 0 now
    reasoning: str

    @field_serializer("aspect", "score", "reasoning", when_used="json")
    def serialize_fields(self, value):
        return value


# -----------------------------
# Signature
# -----------------------------
class ParentCompanyRelationshipScoring(dspy.Signature):
    """
    You are a corporate governance risk expert.

    Evaluate Parent Company Relationship risk.
    Score each aspect from 0 (no risk) to 5 (very high risk).

    If no parent company exists → score should be 0.
    """

    governance_text: str = dspy.InputField(
        desc="Governance report text about parent company, controlling shareholder, transactions, independence, director dispatch"
    )

    parent_existence: ScoringAspect = dspy.OutputField(
        desc="Risk from existence of a parent company (0-5)"
    )

    capital_relationship: ScoringAspect = dspy.OutputField(
        desc="Risk from ownership ratio, voting rights (資本関係・持株比率・議決権比率) (0-5)"
    )

    controlling_shareholder: ScoringAspect = dspy.OutputField(
        desc="Risk from controlling shareholder presence (支配株主) (0-5)"
    )

    related_party_transactions: ScoringAspect = dspy.OutputField(
        desc="Risk from transactions with parent / related parties (0-5)"
    )

    management_independence: ScoringAspect = dspy.OutputField(
        desc="Risk to management independence (経営の独立性) (0-5)"
    )

    director_dispatch: ScoringAspect = dspy.OutputField(
        desc="Risk from director/employee dispatch (役員派遣) (0-5)"
    )


# -----------------------------
# Module
# -----------------------------
class ParentCompanyRelationshipModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(ParentCompanyRelationshipScoring, n=6)

    def forward(self, governance_text: str):

        # ---------- RULE: No Parent Company ----------
        if "親会社の有無" in governance_text and "なし" in governance_text:
            result = {
                "parent_company_risk_score": 0.0,
                "aspects": [
                    {
                        "aspect": "Parent Company Existence",
                        "score": 0,
                        "reasoning": "The governance report explicitly states 親会社の有無: なし (no parent company). Therefore, parent company relationship risk is zero."
                    },
                    {
                        "aspect": "Capital Relationship",
                        "score": 0,
                        "reasoning": "No parent company means no ownership/voting control risk from a parent entity."
                    },
                    {
                        "aspect": "Controlling Shareholder",
                        "score": 0,
                        "reasoning": "No parent company implies no parent-driven controlling shareholder risk."
                    },
                    {
                        "aspect": "Related Party Transactions",
                        "score": 0,
                        "reasoning": "Without a parent company, there are no parent-related transactions creating governance risk."
                    },
                    {
                        "aspect": "Management Independence",
                        "score": 0,
                        "reasoning": "Management independence is structurally preserved because no parent influence exists."
                    },
                    {
                        "aspect": "Director Dispatch",
                        "score": 0,
                        "reasoning": "No parent company means no director/employee dispatch risk."
                    }
                ],
                "conclusion": "No parent company relationship risk"
            }
            return result

        # ---------- Normal scoring if parent exists ----------
        scoring = self.scorer(governance_text=governance_text)

        aspects = [
            scoring.parent_existence,
            scoring.capital_relationship,
            scoring.controlling_shareholder,
            scoring.related_party_transactions,
            scoring.management_independence,
            scoring.director_dispatch,
        ]

        # Risk weights
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]

        parent_company_risk_score = round(
            sum(a.score * w for a, w in zip(aspects, weights)), 2
        )

        result = {
            "parent_company_risk_score": parent_company_risk_score,
            "aspects": [a.model_dump() for a in aspects],
            "conclusion": self._label_score(parent_company_risk_score),
        }

        return result

    @staticmethod
    def _label_score(score: float) -> str:
        if score == 0:
            return "No parent company relationship risk"
        elif score <= 1.5:
            return "Low parent influence risk"
        elif score <= 3.0:
            return "Moderate parent governance risk"
        elif score <= 4.5:
            return "High parent control risk"
        else:
            return "Severe parent control & minority shareholder risk"


# -----------------------------
# Usage Example
# -----------------------------
parent_company_scorer = ParentCompanyRelationshipModule()

text = """
The Company's Board of Directors is composed of six directors, four of whom are independent outside directors, ensuring independence. In addition, as an advisory body to the Board of Directors, the Company establishes a Governance Committee composed of said four independent outside directors. The aforementioned Committee conducts deliberations on transactions between the Company and related parties such as SoftBank Group Corp., SoftBank Corp., A Holdings Corporation, NAVER Corporation, and their subsidiaries (the "Related Party Transactions") from the perspectives of fairness, economic rationality, and legality. Furthermore, the division responsible for governance conducts a review of Related Party Transactions that do not require deliberation by the Governance Committee. For transactions meeting specific criteria, the independent outside director serving as a full-time Audit and Supervisory Committee member is authorized by the Governance Committee to conduct a prior review from the same perspectives as the Governance Committee.

Name of Parent Company, if applicable | SoftBank Corp. (Listed Stock Exchange: Tokyo (Code: 9434))
A Holdings Corporation | 4,467,326,675 | 62.44
A Holdings Corporation, the largest shareholder, is a subsidiary of SoftBank Corp., and SoftBank Corp. is the parent company, etc. that has the greatest influence on the Company.

4. Policy on Measures to Protect Minority Shareholders in Conducting Transactions with Controlling Shareholder
The Company's Board of Directors is composed of six directors, four of whom are independent outside directors, ensuring independence. In addition, as an advisory body to the Board of Directors, the Company establishes a Governance Committee composed of said four independent outside directors. The aforementioned Committee conducts deliberations on Related Party Transactions from the perspectives of fairness, economic rationality, and legality.
Furthermore, the division responsible for governance conducts a review of Related Party Transactions that do not require deliberation by the Governance Committee. For transactions meeting specific criteria, the independent outside director serving as a full-time Audit and Supervisory Committee member is authorized by the Governance Committee to conduct a prior review from the same perspectives as the Governance Committee.
Under the goal of maximizing the Group's value, the Group respects the autonomy of its Group companies and ensures their independence to work together to create synergies and to continuously evolve and grow.
However, the Company refrains from imposing prior approval requirements on its listed subsidiaries and affiliates that would affect their independence, and gives consideration not to unduly constrain decision-making of each company.

However, the Company refrains from imposing prior approval requirements on its listed subsidiaries and affiliates that would affect their independence, and gives consideration not to unduly constrain decision-making of each company.
Furthermore, the Company's outside directors regularly meet with the outside directors, etc. of individual listed subsidiaries to confirm that the Company is not unfairly restraining the decision-making of each company.
Autonomous business management that respects the interest of their minority shareholders.
While maintaining independence and autonomy as a listed subsidiary, the Company believes that ASKUL's growth and continued pursuit of Group synergies will contribute to the enhancement of the corporate value of the company and the Group as a whole.

(4) Ensuring independence from the parent company, etc.
(i) Parent company's approach and policies regarding group management
Please refer to the Corporate Governance Report of SoftBank Corp. for SoftBank Corp.'s approach and policies on group management.

(ii) Approach and measures to ensure independence from the parent company that are necessary to protect minority shareholders
There are no directors of the Company who concurrently serve as directors of the parent company, and there are no directors invited from the parent company who concurrently serve as directors or employees of the parent company. Also, the Company relies very little on its parent company or other members of its company group for its business transactions. Most of its partners in its transactions are consumers or corporations with no investment relationship with the Company.

Furthermore, the Company has enacted "Regulations for Appropriate Business Transactions and Practices by LY Corporation, its Parent Company, Subsidiaries, and Affiliates." In these regulations, the Company has intentionally and expressly stipulated the prohibition of: transactions with the parent company which are clearly advantageous or disadvantageous compared to transactions with third parties or to comparable transactions; and transactions for the purpose of shifting profits, losses, or risks.

The Company has stipulated in the Regulations of the Board of Directors Meetings that a person having a special interest in the resolution of the Board of Directors cannot exercise his/her voting rights. In addition, the Company endeavors to make an accurate judgment when determining whether a person falls under a person having a special interest by seeking advice of external experts as necessary.

The Company’s Board of Directors is composed of six directors, four of whom are independent outside directors, ensuring independence. In addition, as an advisory body to the Board of Directors, the Company establishes a Governance Committee composed of said four independent outside directors. The aforementioned Committee conducts deliberations on Related Party Transactions, monitoring decision-making as they are carried out, from the perspectives of fairness, economic rationality, and legality.

(iii) Agreement related to parent company's approach and policies regarding group management
The Company entered into a capital alliance agreement with A Holdings Corporation on December 23, 2019, primarily to create synergies through business integration across the Company's various business domains. Subsequently, due to changes in the composition of the Board of Directors, a memorandum of amendment was executed on May 16, 2025.

As this agreement and the memorandum of amendment are contracts with the Company's parent company, careful deliberations were conducted by the Governance Committee, which serves as an advisory body to the Board of Directors and is composed of independent outside directors, to ensure that the interests of all shareholders, including minority shareholders, are not harmed.

The agreement was concluded following deliberation and resolution by the Board of Directors. For an overview of the agreement, please refer to the Company's securities report.

b. Person who executes business for a non-executive director of a parent company
"""

result = parent_company_scorer(text)

print(json.dumps(result, indent=2, ensure_ascii=False))