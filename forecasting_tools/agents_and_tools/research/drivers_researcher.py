from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.helpers.metaculus_api import MetaculusQuestion
from forecasting_tools.util import async_batching
from forecasting_tools.util.misc import clean_indents

if TYPE_CHECKING:
    from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
        BaseRateReport,
    )

logger = logging.getLogger(__name__)


class DriversResearcher:

    PRECONDITION_THRESHOLD = 0.2

    @classmethod
    async def research_drivers(
        cls,
        metaculus_question: MetaculusQuestion,
        num_drivers_to_return: int = 8,
        broad_scan_model: str = "openrouter/anthropic/claude-sonnet-4.5",
        base_rate_context: "BaseRateReport | None" = None,
    ) -> list[ScoredDriver]:
        question_details = metaculus_question.give_question_details_as_markdown()

        candidates = await cls._broad_scan(question_details, broad_scan_model)
        logger.info(f"Phase 1: Generated {len(candidates)} candidate drivers")

        num_to_filter_to = min(len(candidates), num_drivers_to_return * 2)
        filtered = await cls._filter_candidates(
            question_details, candidates, num_to_filter_to
        )
        logger.info(f"Phase 1: Filtered to {len(filtered)} candidates")

        logger.info(f"Starting precondition validation for {len(filtered)} candidates")
        validated_candidates, precondition_analyses = (
            await cls._precondition_validation(
                question_details, filtered, base_rate_context
            )
        )
        logger.info(
            f"Precondition validation complete. {len(validated_candidates)} drivers above threshold."
        )

        scored = await cls._validate_with_search(question_details, validated_candidates)
        logger.info(f"Phase 2: {len(scored)} drivers validated with evidence")

        if len(scored) > num_drivers_to_return:
            scored = await cls._final_selection(
                question_details, scored, num_drivers_to_return
            )

        return scored

    @classmethod
    async def _search_and_extract(
        cls,
        search_query: str,
        extraction_prompt: str,
        return_type: type,
    ) -> object:
        news_context = await AskNewsSearcher().get_formatted_news_async(
            search_query
        )
        llm = GeneralLlm(
            model="openrouter/anthropic/claude-sonnet-4.5",
            temperature=0,
        )
        full_prompt = (
            f"{extraction_prompt}\n\nNews context:\n{news_context}"
        )
        return await llm.invoke_and_return_verified_type(
            full_prompt, return_type
        )

    @classmethod
    async def _broad_scan(
        cls,
        question_details: str,
        model: str = "openrouter/anthropic/claude-sonnet-4.5",
    ) -> list[CandidateDriver]:
        prompt = clean_indents(
            f"""
            You are a horizon-scanning analyst performing a STEEP analysis
            (Social, Technological, Economic, Environmental, Political) for
            a forecasting question.

            Question:
            {question_details}

            Brainstorm 30-50 candidate drivers that could influence the outcome
            of this question. For each driver, identify:
            - A short name
            - Which STEEP category it belongs to
            - The mechanism by which it influences the outcome
            - Whether it is accelerating, decelerating, stable, or unclear
              in its effect on the outcome
            - An initial relevance score from 0.0 to 1.0

            Cast a wide net across all five STEEP categories.
            Return your answer as a JSON list.
            """
        )
        llm = GeneralLlm(model=model, temperature=0.7)
        candidates = await llm.invoke_and_return_verified_type(
            prompt, list[CandidateDriver]
        )
        return candidates

    @classmethod
    async def _filter_candidates(
        cls,
        question_details: str,
        candidates: list[CandidateDriver],
        num_to_keep: int,
    ) -> list[CandidateDriver]:
        candidate_list = "\n".join(
            f"{i}. [{c.category.value}] {c.name} (relevance: {c.initial_relevance}) - {c.mechanism}"
            for i, c in enumerate(candidates)
        )
        prompt = clean_indents(
            f"""
            You are a forecasting analyst. Given the question and candidate
            drivers below, select the {num_to_keep} most relevant and diverse
            drivers. Prefer drivers that are specific, actionable, and span
            multiple STEEP categories.

            Question:
            {question_details}

            Candidates:
            {candidate_list}

            Return only a JSON list of integers corresponding to the selected
            candidates, in order of relevance.
            """
        )
        llm = GeneralLlm(model="openrouter/anthropic/claude-opus-4.6", temperature=0)
        indices = await llm.invoke_and_return_verified_type(prompt, list[int])
        return [candidates[i] for i in indices if 0 <= i < len(candidates)]

    @classmethod
    async def _precondition_validation(
        cls,
        question_details: str,
        candidates: list[CandidateDriver],
        base_rate_context: "BaseRateReport | None" = None,
    ) -> tuple[list[CandidateDriver], list[PreconditionAnalysis]]:
        """
        Validates drivers by analyzing dominance scenarios and preconditions.

        Phase 1: Generate dominance scenarios for each driver (parallel)
        Phase 2: Identify preconditions for each driver (parallel)
        Phase 3: Search for evidence on preconditions (rate-limited parallel)
        Phase 4: Re-rank based on precondition alignment scores
        """
        if not candidates:
            return [], []

        base_rate_str = ""
        if base_rate_context:
            base_rate_str = f"\nHistorical base rate context: {base_rate_context.markdown_report[:500]}..."

        # Phase 1 & 2: Get dominance scenarios and preconditions (parallel per driver)
        async def analyze_driver(
            candidate: CandidateDriver,
        ) -> tuple[CandidateDriver, DominanceScenario, list[Precondition]] | None:
            llm = GeneralLlm(
                model="openrouter/anthropic/claude-opus-4.6",
                temperature=0.3,
            )

            # Phase 1: Dominance scenario
            dominance_prompt = clean_indents(
                f"""
                You are analyzing a driver for a forecasting question.

                Driver: {candidate.name}
                Mechanism: {candidate.mechanism}
                Category: {candidate.category.value}

                Question:
                {question_details}
                {base_rate_str}

                Imagine this driver becomes THE DOMINANT FORCE shaping the outcome
                of this question.

                1. What would that scenario look like in concrete terms?
                2. How plausible is this within the question's timescale?
                   (Answer: high, medium, low, or very_low)
                3. What would be the key ripple effects on related systems?

                Return your answer as a JSON object with keys:
                - scenario_description: A 2-3 sentence description
                - timescale_plausibility: One of high/medium/low/very_low
                - system_effects: A list of 2-4 key effects
                """
            )
            try:
                dominance = await llm.invoke_and_return_verified_type(
                    dominance_prompt, DominanceScenario
                )
            except Exception:
                logger.warning(f"Dominance analysis failed for: {candidate.name}")
                return None

            # Phase 2: Preconditions
            preconditions_prompt = clean_indents(
                f"""
                You are analyzing what preconditions would need to be true for a
                driver to become dominant in affecting a forecasting question.

                Driver: {candidate.name}
                Mechanism: {candidate.mechanism}
                Dominance scenario: {dominance.scenario_description}

                Question:
                {question_details}

                For this driver to become dominant:
                - What 3-6 specific, observable preconditions would need to be true?
                - Focus on things we can search for in current news/data
                - Be concrete and specific

                Return your answer as a JSON list of objects with keys:
                - description: What the precondition is
                - why_necessary: Why this is needed for the driver to dominate
                """
            )
            try:
                preconditions = await llm.invoke_and_return_verified_type(
                    preconditions_prompt, list[Precondition]
                )
            except Exception:
                logger.warning(f"Preconditions analysis failed for: {candidate.name}")
                return None

            return (candidate, dominance, preconditions)

        coroutines = [analyze_driver(c) for c in candidates]
        results, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                coroutines
            )
        )
        driver_analyses = [r for r in results if r is not None]

        if not driver_analyses:
            return candidates, []

        # Phase 3: Search for precondition evidence (rate-limited)
        all_search_tasks = []
        for idx, (candidate, dominance, preconditions) in enumerate(driver_analyses):
            for precondition in preconditions:
                all_search_tasks.append((idx, candidate, dominance, precondition))

        async def search_precondition(
            task: tuple[int, CandidateDriver, DominanceScenario, Precondition],
        ) -> tuple[int, CandidateDriver, DominanceScenario, Precondition]:
            idx, candidate, dominance, precondition = task
            extraction_prompt = clean_indents(
                f"""
                Analyze the following news context for evidence about a
                precondition for a forecasting analysis.

                Precondition: {precondition.description}
                Context: Driver "{candidate.name}" for question about
                {question_details[:200]}...

                Determine if this precondition is:
                - emerging: Evidence shows active development toward this
                - stable: Exists but not significantly changing
                - absent: No evidence this is happening
                - contrary: Evidence shows movement away from this

                Return your findings as a JSON object with keys:
                - status: One of emerging/stable/absent/contrary
                - evidence_summary: 1-2 sentence summary of what you found
                - citations: List of citation strings
                """
            )
            try:
                result = await cls._search_and_extract(
                    search_query=f"{candidate.name} {precondition.description}",
                    extraction_prompt=extraction_prompt,
                    return_type=dict,
                )
                precondition.status = PreconditionStatus(result.get("status", "absent"))
                precondition.evidence_summary = result.get("evidence_summary", "")
                precondition.citations = result.get("citations", [])
            except Exception:
                logger.warning(
                    f"Precondition search failed for: {precondition.description[:50]}"
                )
                precondition.status = PreconditionStatus.ABSENT

            return (idx, candidate, dominance, precondition)

        # Rate limit: 5 requests per second (300 per minute)
        rate_limited_coroutines = async_batching.wrap_coroutines_with_rate_limit(
            [search_precondition(t) for t in all_search_tasks],
            calls_per_period=300,
            time_period_in_seconds=60,
        )
        search_results, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                rate_limited_coroutines
            )
        )

        # Phase 4: Re-rank based on precondition alignment
        # Group preconditions back by driver (keyed by index to avoid name collisions)
        driver_preconditions: dict[int, list[Precondition]] = {}
        driver_dominance: dict[int, DominanceScenario] = {}
        driver_candidate: dict[int, CandidateDriver] = {}
        for idx, (candidate, dominance, preconditions) in enumerate(driver_analyses):
            driver_preconditions[idx] = preconditions
            driver_dominance[idx] = dominance
            driver_candidate[idx] = candidate

        # Update with search results
        for idx, candidate, dominance, precondition in search_results:
            # Find and update the precondition in our dict
            for p in driver_preconditions.get(idx, []):
                if p.description == precondition.description:
                    p.status = precondition.status
                    p.evidence_summary = precondition.evidence_summary
                    p.citations = precondition.citations

        # Calculate scores and create analyses
        analyses: list[PreconditionAnalysis] = []
        scored_candidates: list[tuple[CandidateDriver, float]] = []

        plausibility_scores = {"high": 1.0, "medium": 0.7, "low": 0.4, "very_low": 0.1}

        for idx, preconditions in driver_preconditions.items():
            candidate = driver_candidate[idx]
            dominance = driver_dominance[idx]

            # Calculate alignment score
            favorable_statuses = {
                PreconditionStatus.EMERGING,
                PreconditionStatus.STABLE,
            }
            favorable_count = sum(
                1 for p in preconditions if p.status in favorable_statuses
            )
            alignment_score = (
                favorable_count / len(preconditions) if preconditions else 0
            )

            # Determine emergence strength
            if alignment_score >= 0.75:
                emergence_strength = "strong"
            elif alignment_score >= 0.5:
                emergence_strength = "moderate"
            elif alignment_score >= 0.25:
                emergence_strength = "weak"
            else:
                emergence_strength = "very_weak"

            analysis = PreconditionAnalysis(
                driver_name=candidate.name,
                dominance_scenario=dominance,
                preconditions=preconditions,
                precondition_alignment_score=alignment_score,
                overall_emergence_strength=emergence_strength,
            )
            analyses.append(analysis)

            # Combined score formula from plan:
            # 0.30 * initial_relevance + 0.40 * alignment + 0.20 * plausibility + 0.10 * evidence_quality
            plausibility = plausibility_scores.get(
                dominance.timescale_plausibility, 0.5
            )
            evidence_quality = sum(1 for p in preconditions if p.citations) / max(
                len(preconditions), 1
            )

            combined_score = (
                0.30 * candidate.initial_relevance
                + 0.40 * alignment_score
                + 0.20 * plausibility
                + 0.10 * evidence_quality
            )
            scored_candidates.append((candidate, combined_score))

        # Filter by threshold and sort
        filtered = [
            (c, s) for c, s in scored_candidates if s >= cls.PRECONDITION_THRESHOLD
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        validated = [c for c, _ in filtered]

        return validated, analyses

    @classmethod
    async def _validate_with_search(
        cls,
        question_details: str,
        candidates: list[CandidateDriver],
    ) -> list[ScoredDriver]:
        async def validate_one(candidate: CandidateDriver) -> ScoredDriver | None:
            extraction_prompt = clean_indents(
                f"""
                Analyze the following news context for evidence about a
                driver and its potential impact on a forecasting question.

                Driver: {candidate.name}
                Category: {candidate.category.value}
                Mechanism: {candidate.mechanism}

                Question context:
                {question_details}

                For each piece of evidence found, provide:
                - A brief summary of the evidence
                - The citation
                - How recent the evidence is (e.g. "2024", "last month", etc.)

                Return your findings as a JSON list of objects with keys:
                "summary", "citation", "recency".
                If no relevant evidence is found, return an empty list.
                """
            )
            try:
                signals = await cls._search_and_extract(
                    search_query=f"{candidate.name} {candidate.mechanism}",
                    extraction_prompt=extraction_prompt,
                    return_type=list[SignalEvidence],
                )
            except Exception:
                logger.warning(f"Search failed for driver: {candidate.name}")
                return None

            if not signals:
                return None

            return ScoredDriver(
                name=candidate.name,
                category=candidate.category,
                mechanism=candidate.mechanism,
                directionality=candidate.directionality,
                signals=signals,
                direction_of_pressure="",
                strength=DriverStrength.MODERATE,
                uncertainty="",
            )

        coroutines = [validate_one(c) for c in candidates]
        results, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                coroutines
            )
        )
        scored = [r for r in results if r is not None]

        if scored:
            scored = await cls._score_drivers(question_details, scored)

        return scored

    @classmethod
    async def _score_drivers(
        cls,
        question_details: str,
        drivers: list[ScoredDriver],
    ) -> list[ScoredDriver]:
        driver_summaries = "\n\n".join(
            f"Driver {i}: {d.name} [{d.category.value}]\n"
            f"Mechanism: {d.mechanism}\n"
            f"Evidence: {'; '.join(s.summary for s in d.signals)}"
            for i, d in enumerate(drivers)
        )
        prompt = clean_indents(
            f"""
            You are a forecasting analyst. For each driver below, assess:
            1. direction_of_pressure: How does this driver push the question
               outcome? (e.g. "pushes toward Yes", "increases the value", etc.)
            2. strength: WEAK, MODERATE, or STRONG
            3. uncertainty: A brief note on how certain we are about this
               driver's effect.

            Question:
            {question_details}

            Drivers:
            {driver_summaries}

            Return a JSON list of objects with keys: "index", "direction_of_pressure",
            "strength", "uncertainty". One object per driver.
            """
        )
        llm = GeneralLlm(model="openrouter/anthropic/claude-opus-4.6", temperature=0)
        assessments = await llm.invoke_and_return_verified_type(
            prompt, list[DriverAssessment]
        )

        for assessment in assessments:
            if 0 <= assessment.index < len(drivers):
                d = drivers[assessment.index]
                d.direction_of_pressure = assessment.direction_of_pressure
                d.strength = assessment.strength
                d.uncertainty = assessment.uncertainty

        return drivers

    @classmethod
    async def _final_selection(
        cls,
        question_details: str,
        drivers: list[ScoredDriver],
        num_to_return: int,
    ) -> list[ScoredDriver]:
        driver_list = "\n".join(
            f"{i}. [{d.category.value}] {d.name} ({d.strength.value}) - {d.direction_of_pressure}"
            for i, d in enumerate(drivers)
        )
        prompt = clean_indents(
            f"""
            Select the {num_to_return} most important and diverse drivers
            from the list below for forecasting this question. Prefer a mix
            of STEEP categories and both directions of pressure.

            Question:
            {question_details}

            Drivers:
            {driver_list}

            Return only a JSON list of integers.
            """
        )
        llm = GeneralLlm(model="openrouter/anthropic/claude-opus-4.6", temperature=0)
        indices = await llm.invoke_and_return_verified_type(prompt, list[int])
        return [drivers[i] for i in indices if 0 <= i < len(drivers)]


class SteepCategory(str, Enum):
    SOCIAL = "social"
    TECHNOLOGICAL = "technological"
    ECONOMIC = "economic"
    ENVIRONMENTAL = "environmental"
    POLITICAL = "political"


class Directionality(str, Enum):
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    STABLE = "stable"
    UNCLEAR = "unclear"


class DriverStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class PreconditionStatus(str, Enum):
    EMERGING = "emerging"  # Evidence shows active development
    STABLE = "stable"  # Exists but not changing
    ABSENT = "absent"  # No evidence
    CONTRARY = "contrary"  # Evidence shows movement away


class Precondition(BaseModel):
    description: str
    why_necessary: str
    status: PreconditionStatus | None = None
    evidence_summary: str | None = None
    citations: list[str] = Field(default_factory=list)


class DominanceScenario(BaseModel):
    scenario_description: str
    timescale_plausibility: str  # high/medium/low/very_low
    system_effects: list[str]


class PreconditionAnalysis(BaseModel):
    driver_name: str
    dominance_scenario: DominanceScenario
    preconditions: list[Precondition]
    precondition_alignment_score: float = Field(ge=0.0, le=1.0)
    overall_emergence_strength: str  # strong/moderate/weak/very_weak


class CandidateDriver(BaseModel):
    name: str
    category: SteepCategory
    mechanism: str
    directionality: Directionality
    initial_relevance: float = Field(ge=0.0, le=1.0)


class SignalEvidence(BaseModel):
    summary: str
    citation: str
    recency: str | None = None


class ScoredDriver(BaseModel):
    name: str
    category: SteepCategory
    mechanism: str
    directionality: Directionality
    signals: list[SignalEvidence]
    direction_of_pressure: str
    strength: DriverStrength
    uncertainty: str

    @property
    def display_text(self) -> str:
        signal_text = "; ".join(s.summary for s in self.signals[:2])
        return (
            f"**{self.name}** [{self.category.value.title()}] "
            f"({self.strength.value}, {self.directionality.value}): "
            f"{self.mechanism}. {self.direction_of_pressure}. "
            f"Evidence: {signal_text}"
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
