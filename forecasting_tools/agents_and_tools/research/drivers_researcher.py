from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.metaculus_api import MetaculusQuestion
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class DriversResearcher:

    @classmethod
    async def research_drivers(
        cls,
        metaculus_question: MetaculusQuestion,
        num_drivers_to_return: int = 8,
        broad_scan_model: str = "openrouter/anthropic/claude-sonnet-4.5",
    ) -> list[ScoredDriver]:
        question_details = metaculus_question.give_question_details_as_markdown()

        # Phase A1: Broad scan + filter (1 Sonnet call)
        candidates = await cls._broad_scan_and_filter(
            question_details, num_drivers_to_return, broad_scan_model
        )
        logger.info(
            f"Phase A1: Generated and filtered to {len(candidates)} candidate drivers"
        )

        # Phase A2: LLM precondition validation (1 Opus call)
        validated = await cls._llm_precondition_validation(
            question_details, candidates
        )
        logger.info(
            f"Phase A2: {len(validated)} drivers passed precondition validation"
        )

        # Phase A3: Score and select (1 Opus call)
        scored = await cls._score_and_select(
            question_details, validated, num_drivers_to_return
        )
        logger.info(f"Phase A3: Final {len(scored)} scored drivers")

        return scored

    @classmethod
    async def _broad_scan_and_filter(
        cls,
        question_details: str,
        num_to_return: int,
        model: str = "openrouter/anthropic/claude-sonnet-4.5",
    ) -> list[CandidateDriver]:
        num_to_keep = min(num_to_return * 2, 16)
        prompt = clean_indents(
            f"""
            You are a horizon-scanning analyst performing a STEEP analysis
            (Social, Technological, Economic, Environmental, Political) for
            a forecasting question.

            Question:
            {question_details}

            Brainstorm 30-50 candidate drivers that could influence the outcome
            of this question. Then select the top {num_to_keep} most relevant
            and diverse drivers. Prefer drivers that are specific, actionable,
            and span multiple STEEP categories.

            For each selected driver, provide:
            - A short name
            - Which STEEP category it belongs to
            - The mechanism by which it influences the outcome
            - Whether it is accelerating, decelerating, stable, or unclear
              in its effect on the outcome
            - A relevance score from 0.0 to 1.0

            Return only the top {num_to_keep} as a JSON list.
            """
        )
        llm = GeneralLlm(model=model, temperature=0.7)
        candidates = await llm.invoke_and_return_verified_type(
            prompt, list[CandidateDriver]
        )
        return candidates

    @classmethod
    async def _llm_precondition_validation(
        cls,
        question_details: str,
        candidates: list[CandidateDriver],
    ) -> list[CandidateDriver]:
        if not candidates:
            return []

        candidate_list = "\n".join(
            f"{i}. [{c.category.value}] {c.name} "
            f"(relevance: {c.initial_relevance}, {c.directionality.value})\n"
            f"   Mechanism: {c.mechanism}"
            for i, c in enumerate(candidates)
        )

        prompt = clean_indents(
            f"""
            You are a forecasting analyst validating candidate drivers for a
            forecasting question. For each candidate driver, assess:

            1. **Dominance plausibility**: How plausible is it that this driver
               becomes the dominant force shaping the outcome within the
               question's timescale? (high/medium/low/very_low)

            2. **Precondition status**: What key preconditions would need to be
               true for this driver to matter? Based on your knowledge, are
               those preconditions currently being met? (emerging/stable/absent/contrary)

            3. **Overall viability score**: On a scale of 0.0 to 1.0, how viable
               is this driver as a significant force on the question outcome?
               Consider relevance, plausibility, and precondition alignment.

            Question:
            {question_details}

            Candidate Drivers:
            {candidate_list}

            For each candidate, return a JSON object with:
            - "index": the candidate number
            - "dominance_plausibility": one of high/medium/low/very_low
            - "precondition_summary": brief assessment of key preconditions
            - "viability_score": float from 0.0 to 1.0

            Return a JSON list of these objects. Filter out any candidates with
            viability_score below 0.3 — do not include them in the output.
            """
        )

        llm = GeneralLlm(
            model="openrouter/anthropic/claude-opus-4.6", temperature=0.3
        )
        assessments = await llm.invoke_and_return_verified_type(
            prompt, list[PreconditionAssessment]
        )

        validated = []
        for assessment in assessments:
            if (
                0 <= assessment.index < len(candidates)
                and assessment.viability_score >= 0.3
            ):
                validated.append(candidates[assessment.index])

        return validated

    @classmethod
    async def _score_and_select(
        cls,
        question_details: str,
        candidates: list[CandidateDriver],
        num_to_return: int,
    ) -> list[ScoredDriver]:
        if not candidates:
            return []

        candidate_list = "\n".join(
            f"{i}. [{c.category.value}] {c.name} ({c.directionality.value})\n"
            f"   Mechanism: {c.mechanism}"
            for i, c in enumerate(candidates)
        )

        prompt = clean_indents(
            f"""
            You are a forecasting analyst. For each driver below, assess:

            1. **direction_of_pressure**: How does this driver push the question
               outcome? (e.g. "pushes toward Yes", "increases the value", etc.)
            2. **strength**: WEAK, MODERATE, or STRONG
            3. **uncertainty**: A brief note on how certain we are about this
               driver's effect.

            Then select the {num_to_return} most important and diverse drivers.
            Prefer a mix of STEEP categories and both directions of pressure.

            Question:
            {question_details}

            Drivers:
            {candidate_list}

            Return a JSON list of objects with keys:
            "index", "direction_of_pressure", "strength", "uncertainty".
            Include only the top {num_to_return} drivers, in order of importance.
            """
        )

        llm = GeneralLlm(
            model="openrouter/anthropic/claude-opus-4.6", temperature=0
        )
        assessments = await llm.invoke_and_return_verified_type(
            prompt, list[DriverAssessment]
        )

        scored_drivers = []
        for assessment in assessments:
            if 0 <= assessment.index < len(candidates):
                c = candidates[assessment.index]
                scored_drivers.append(
                    ScoredDriver(
                        name=c.name,
                        category=c.category,
                        mechanism=c.mechanism,
                        directionality=c.directionality,
                        direction_of_pressure=assessment.direction_of_pressure,
                        strength=assessment.strength,
                        uncertainty=assessment.uncertainty,
                    )
                )

        return scored_drivers[:num_to_return]


class SteepCategory(str, Enum):
    SOCIAL = "social"
    TECHNOLOGICAL = "technological"
    ECONOMIC = "economic"
    ENVIRONMENTAL = "environmental"
    POLITICAL = "political"

    @classmethod
    def _missing_(cls, value: object) -> SteepCategory | None:
        if isinstance(value, str):
            lowered = value.lower()
            for member in cls:
                if member.value == lowered:
                    return member
        return None


class Directionality(str, Enum):
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    STABLE = "stable"
    UNCLEAR = "unclear"

    @classmethod
    def _missing_(cls, value: object) -> Directionality | None:
        if isinstance(value, str):
            lowered = value.lower()
            for member in cls:
                if member.value == lowered:
                    return member
        return None


class DriverStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

    @classmethod
    def _missing_(cls, value: object) -> DriverStrength | None:
        if isinstance(value, str):
            lowered = value.lower()
            for member in cls:
                if member.value == lowered:
                    return member
        return None


class PreconditionStatus(str, Enum):
    EMERGING = "emerging"
    STABLE = "stable"
    ABSENT = "absent"
    CONTRARY = "contrary"


class Precondition(BaseModel):
    description: str
    why_necessary: str
    status: PreconditionStatus | None = None
    evidence_summary: str | None = None
    citations: list[str] = Field(default_factory=list)


class DominanceScenario(BaseModel):
    scenario_description: str
    timescale_plausibility: str
    system_effects: list[str]


class PreconditionAnalysis(BaseModel):
    driver_name: str
    dominance_scenario: DominanceScenario
    preconditions: list[Precondition]
    precondition_alignment_score: float = Field(ge=0.0, le=1.0)
    overall_emergence_strength: str


class PreconditionAssessment(BaseModel):
    index: int
    dominance_plausibility: str
    precondition_summary: str
    viability_score: float = Field(ge=0.0, le=1.0)


class CandidateDriver(BaseModel):
    model_config = {"populate_by_name": True}

    name: str
    category: SteepCategory
    mechanism: str
    directionality: Directionality = Field(validation_alias="trend")
    initial_relevance: float = Field(ge=0.0, le=1.0, validation_alias="relevance")

    @field_validator("directionality", mode="before")
    @classmethod
    def _coerce_directionality(cls, v: object) -> object:
        if isinstance(v, str):
            return v.lower()
        return v


class SignalEvidence(BaseModel):
    summary: str
    citation: str
    recency: str | None = None


class ScoredDriver(BaseModel):
    name: str
    category: SteepCategory
    mechanism: str
    directionality: Directionality
    direction_of_pressure: str
    strength: DriverStrength
    uncertainty: str

    @property
    def display_text(self) -> str:
        return (
            f"**{self.name}** [{self.category.value.title()}] "
            f"({self.strength.value}, {self.directionality.value}): "
            f"{self.mechanism}. {self.direction_of_pressure}."
        )

    @classmethod
    def turn_drivers_into_markdown(cls, drivers: list[ScoredDriver]) -> str:
        if not drivers:
            return "No drivers identified."
        lines: list[str] = []
        for driver in drivers:
            lines.append(f"- {driver.display_text}")
        return "\n".join(lines)


class DriverAssessment(BaseModel):
    index: int
    direction_of_pressure: str
    strength: DriverStrength
    uncertainty: str
