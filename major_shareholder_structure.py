import dspy
from pydantic import BaseModel, field_serializer
from typing import Literal

lm = dspy.LM(
    model="openai/gemini-2.5-flash",
    api_base="http://localhost:8317/v1",
    api_key="proxypal-local",
)

dspy.configure(lm=lm)


class ScoringAspect(BaseModel):
    """The detail on the scoring for the aspect"""

    aspect: str
    score: Literal[1, 2, 3, 4, 5]
    reasoning: str

    @field_serializer("aspect", "score", "reasoning", when_used="json")
    def serialize_optional_fields(self, value):
        return value


class MajorShareholderScoring(dspy.Signature):
    """
    You are a corporate governance and ESG expert.
    Evaluate the company's Major Shareholder Structure based on ownership data.
    Score each aspect objectively from 1 (very weak) to 5 (very strong),
    and provide clear reasoning based on governance risk and minority shareholder protection.
    """

    shareholder_structure: str = dspy.InputField(
        desc="Information about major shareholders, ownership percentages, treasury shares, and voting rights"
    )

    controlling_shareholder_risk: ScoringAspect = dspy.OutputField(
        desc="Is there a controlling shareholder (>30% voting power or board control)? (1-5)"
    )

    ownership_concentration: ScoringAspect = dspy.OutputField(
        desc="How concentrated is ownership among top shareholders? Is ownership well dispersed? (1-5)"
    )

    shareholder_quality: ScoringAspect = dspy.OutputField(
        desc="Quality of major shareholders (institutional vs family/state/parent company). (1-5)"
    )

    minority_shareholder_protection: ScoringAspect = dspy.OutputField(
        desc="Risk to minority shareholders (cross-shareholding, pyramids, unequal voting rights). (1-5)"
    )

    transparency_disclosure: ScoringAspect = dspy.OutputField(
        desc="Clarity and transparency of ownership disclosure and treasury share treatment. (1-5)"
    )


class MajorShareholderScoringModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(MajorShareholderScoring, n=5)

    def forward(self, shareholder_structure: str):
        scoring = self.scorer(shareholder_structure=shareholder_structure)

        aspects = [
            scoring.controlling_shareholder_risk,
            scoring.ownership_concentration,
            scoring.shareholder_quality,
            scoring.minority_shareholder_protection,
            scoring.transparency_disclosure,
        ]

        # Governance-style weights (MSCI-like)
        weights = [0.30, 0.20, 0.25, 0.15, 0.10]

        major_shareholder_score = round(
            sum(a.score * w for a, w in zip(aspects, weights)), 2
        )

        return dspy.Prediction(
            major_shareholder_score=major_shareholder_score,
            aspects=[a.model_dump() for a in aspects],
            conclusion=self._label_score(major_shareholder_score),
        )

    @staticmethod
    def _label_score(score: float) -> str:
        if score >= 4.5:
            return "Very Strong ownership structure"
        elif score >= 3.5:
            return "Healthy / balanced ownership"
        elif score >= 2.5:
            return "Neutral ownership structure"
        elif score >= 1.5:
            return "Weak governance risk present"
        else:
            return "High control & governance risk"


major_shareholder_scorer = MajorShareholderScoringModule()
text = """
## Major Shareholder Structure

**Status of Major Shareholders (as of March 31, 2025):**

| Name or Company Name                                    | Number of Shares Owned   | Percentage (%)   |
|---------------------------------------------------------|--------------------------|------------------|
| A Holdings Corporation                                  | 4,467,326,675            | 62.44            |
| The Master Trust Bank of Japan, Ltd. (Trust Account)    | 508,913,300              | 7.11             |
| STATE STREET BANK AND TRUST COMPANY 505325              | 235,044,681              | 3.29             |
| Custody Bank of Japan, Ltd. (Trust account)             | 208,661,700              | 2.92             |
| STATE STREET BANK AND TRUST COMPANY 505001              | 97,103,019               | 1.36             |
| STATE STREET BANK WEST CLIENT-TREATY 505234             | 56,668,849               | 0.79             |
| STATE STREET BANK AND TRUST COMPANY 505223              | 48,958,854               | 0.68             |
| STATE STREET BANK AND TRUST COMPANY 505103              | 40,108,252               | 0.56             |
| JP MORGAN CHASE BANK 385781                             | 35,824,487               | 0.50             |
| NORTHERN TRUST CO.(AVFC) RE NON TREATY  CLIENTS ACCOUNT | 34,764,681               | 0.49             |

**Controlling Shareholder (excluding Parent Company):** ―

**Parent Company:** SoftBank Corp. (Listed Stock Exchange: Tokyo (Code: 9434))

**Supplementary Explanation:**
*   The Company has treasury stock of 607,074 shares as of March 31, 2025. This treasury stock does not include the Company's shares (28,167,999 shares) held by the Stock Delivery Trust (J-ESOP), RSU Plan (Board Incentive Plan Trust), and Stock Delivery ESOP Trust.
*   A Holdings Corporation, the largest shareholder, is a subsidiary of SoftBank Corp., and SoftBank Corp. is the parent company, etc. that has the greatest influence on the Company.
"""
major_shareholder_scorer(text)