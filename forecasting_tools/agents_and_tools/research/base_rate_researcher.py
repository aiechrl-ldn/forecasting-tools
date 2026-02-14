from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.metaculus_api import MetaculusQuestion
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class LightweightBaseRateResearcher:

    @classmethod
    async def research_base_rates(
        cls,
        question: MetaculusQuestion,
        num_base_rates: int = 3,
        model: str = "openrouter/anthropic/claude-sonnet-4.5",
    ) -> list[BaseRateEstimate]:
        question_details = question.give_question_details_as_markdown()

        prompt = clean_indents(
            f"""
            You are a superforecaster performing base rate analysis for a
            forecasting question. Identify {num_base_rates} relevant reference
            classes and estimate their historical rates.

            For each reference class:
            1. Define a clear numerator and denominator
            2. Estimate the count for each based on your knowledge
            3. Calculate the historical rate as a decimal between 0.0 and 1.0
               (e.g., 40% should be 0.4, NOT 40.0)
            4. Explain why this reference class is relevant

            Choose reference classes that:
            - Are as specific and relevant to the question as possible
            - Span different levels of abstraction (narrow vs broad)
            - Have reasonably well-known historical data

            Question:
            {question_details}

            Return exactly {num_base_rates} reference classes as a JSON list.
            """
        )

        llm = GeneralLlm(model=model, temperature=0.2)
        estimates = await llm.invoke_and_return_verified_type(
            prompt, list[BaseRateEstimate]
        )
        return estimates


class BaseRateEstimate(BaseModel):
    reference_class: str = Field(
        description="e.g., 'US government shutdowns since 1976'"
    )
    numerator_description: str = Field(
        description="e.g., 'Shutdowns lasting >2 weeks'"
    )
    denominator_description: str = Field(
        description="e.g., 'Total government shutdowns'"
    )
    numerator: int
    denominator: int = Field(gt=0)
    historical_rate: float = Field(ge=0.0, le=1.0)
    time_period: str = Field(description="e.g., '1976-2025'")
    relevance_reasoning: str

    @field_validator("historical_rate", mode="before")
    @classmethod
    def normalize_historical_rate(cls, v: float) -> float:
        if v > 1.0:
            return v / 100.0
        return v

    @classmethod
    def format_as_markdown(cls, estimates: list[BaseRateEstimate]) -> str:
        if not estimates:
            return "No base rates identified."
        lines: list[str] = []
        for est in estimates:
            lines.append(
                f"- **{est.reference_class}** ({est.time_period}): "
                f"{est.numerator}/{est.denominator} = "
                f"{est.historical_rate:.0%}. "
                f"{est.relevance_reasoning}"
            )
        return "\n".join(lines)
