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


class ShareholderReturnPolicyScoring(dspy.Signature):
    """
    You are an equity analyst specializing in Japanese listed companies.

    Evaluate the shareholder return policy objectively.
    Scores must follow market norms (1 = weak, 5 = best-in-class).
    """

    policy_text: str = dspy.InputField(
        desc="Official shareholder return policy text disclosed by the company"
    )

    quantitative_commitment: ScoringAspect = dspy.OutputField(
        desc="Numeric commitment: payout ratio, DOE, minimum dividend, formula-based dividend"
    )

    dividend_sustainability: ScoringAspect = dspy.OutputField(
        desc="Dividend stability and predictability: progressive, minimum, multi-year guidance"
    )

    buyback_discipline: ScoringAspect = dspy.OutputField(
        desc="Share buybacks: clear framework, size, timing, rationale"
    )

    capital_allocation_logic: ScoringAspect = dspy.OutputField(
        desc="Clarity of balance between growth investment, shareholder return, and financial soundness"
    )

    governance_alignment: ScoringAspect = dspy.OutputField(
        desc="Alignment with governance and KPIs: ROE, DOE, PBR, board-level decision, mid-term plan"
    )


class ShareholderReturnPolicyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(ShareholderReturnPolicyScoring, n=5)

        self.weights = {
            "quantitative_commitment": 0.25,
            "dividend_sustainability": 0.25,
            "buyback_discipline": 0.20,
            "capital_allocation_logic": 0.20,
            "governance_alignment": 0.10,
        }

    def forward(self, policy_text: str):
        result = self.scorer(policy_text=policy_text)

        aspects = [
            result.quantitative_commitment,
            result.dividend_sustainability,
            result.buyback_discipline,
            result.capital_allocation_logic,
            result.governance_alignment,
        ]

        final_score = round(
            sum(
                getattr(result, name).score * weight
                for name, weight in self.weights.items()
            ),
            2,
        )

        return dspy.Prediction(
            shareholder_return_score=final_score,
            aspects=[a.model_dump() for a in aspects],
        )


text = """当社は常に、①成長投資、②株主還元、③財務改善の3要素のバランスを重視しています。最適なレバレッジをかけて企業が成長すれば、それは株主の皆様への最適な還元にもつながると考えています。

自己株式の取得に関しては、2024年8月にマーケットの急激な変動を踏まえて5,000億円の枠を設定しました。当年度に237,045百万円（28,812,200株）を取得しており、自己株式の取得による支出は2,371億円でした。

2025年度以降は大型投資が控えており、成長投資を優先するモードにあります。
"""

policy_scorer = ShareholderReturnPolicyModule()
result = policy_scorer(text)
result