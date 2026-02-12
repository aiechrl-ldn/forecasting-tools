from unittest.mock import AsyncMock, patch

import pytest

from forecasting_tools.agents_and_tools.research.drivers_researcher import (
    CandidateDriver,
    Directionality,
    DominanceScenario,
    DriverAssessment,
    DriversResearcher,
    DriverStrength,
    Precondition,
    PreconditionAnalysis,
    PreconditionAssessment,
    PreconditionStatus,
    ScoredDriver,
    SignalEvidence,
    SteepCategory,
)


def _make_candidate(
    name: str = "Test Driver",
    category: SteepCategory = SteepCategory.TECHNOLOGICAL,
) -> CandidateDriver:
    return CandidateDriver(
        name=name,
        category=category,
        mechanism="Test mechanism",
        directionality=Directionality.ACCELERATING,
        initial_relevance=0.8,
    )


def _make_scored_driver(
    name: str = "Test Driver",
    category: SteepCategory = SteepCategory.TECHNOLOGICAL,
) -> ScoredDriver:
    return ScoredDriver(
        name=name,
        category=category,
        mechanism="Test mechanism",
        directionality=Directionality.ACCELERATING,
        direction_of_pressure="pushes toward Yes",
        strength=DriverStrength.STRONG,
        uncertainty="Low uncertainty",
    )


class TestPydanticModels:
    def test_candidate_driver_validation(self) -> None:
        driver = _make_candidate()
        assert driver.name == "Test Driver"
        assert driver.initial_relevance == 0.8

    def test_candidate_driver_relevance_bounds(self) -> None:
        with pytest.raises(Exception):
            CandidateDriver(
                name="Bad",
                category=SteepCategory.SOCIAL,
                mechanism="m",
                directionality=Directionality.STABLE,
                initial_relevance=1.5,
            )

    def test_signal_evidence_optional_recency(self) -> None:
        signal = SignalEvidence(summary="s", citation="c")
        assert signal.recency is None

    def test_scored_driver_display_text(self) -> None:
        driver = _make_scored_driver()
        text = driver.display_text
        assert "Test Driver" in text
        assert "Technological" in text
        assert "strong" in text

    def test_turn_drivers_into_markdown_empty(self) -> None:
        result = ScoredDriver.turn_drivers_into_markdown([])
        assert result == "No drivers identified."

    def test_turn_drivers_into_markdown(self) -> None:
        drivers = [
            _make_scored_driver("Driver A", SteepCategory.SOCIAL),
            _make_scored_driver("Driver B", SteepCategory.ECONOMIC),
        ]
        result = ScoredDriver.turn_drivers_into_markdown(drivers)
        assert "Driver A" in result
        assert "Driver B" in result
        assert result.startswith("- ")

    def test_steep_category_values(self) -> None:
        assert len(SteepCategory) == 5

    def test_driver_assessment_model(self) -> None:
        assessment = DriverAssessment(
            index=0,
            direction_of_pressure="pushes toward Yes",
            strength=DriverStrength.WEAK,
            uncertainty="high",
        )
        assert assessment.strength == DriverStrength.WEAK

    def test_precondition_assessment_model(self) -> None:
        assessment = PreconditionAssessment(
            index=0,
            dominance_plausibility="high",
            precondition_summary="Preconditions mostly met",
            viability_score=0.8,
        )
        assert assessment.viability_score == 0.8


class TestDriversResearcher:
    @patch("forecasting_tools.agents_and_tools.research.drivers_researcher.GeneralLlm")
    async def test_research_drivers_success(
        self, mock_llm_cls: AsyncMock
    ) -> None:
        from code_tests.unit_tests.forecasting_test_manager import (
            ForecastingTestManager,
        )

        question = ForecastingTestManager.get_fake_binary_question()

        candidates = [_make_candidate(f"Driver {i}") for i in range(16)]
        precondition_assessments = [
            PreconditionAssessment(
                index=i,
                dominance_plausibility="high",
                precondition_summary="Met",
                viability_score=0.8,
            )
            for i in range(10)
        ]
        driver_assessments = [
            DriverAssessment(
                index=i,
                direction_of_pressure="pushes Yes",
                strength=DriverStrength.STRONG,
                uncertainty="low",
            )
            for i in range(8)
        ]

        mock_llm_instance = AsyncMock()
        mock_llm_instance.invoke_and_return_verified_type = AsyncMock(
            side_effect=[candidates, precondition_assessments, driver_assessments]
        )
        mock_llm_cls.return_value = mock_llm_instance

        result = await DriversResearcher.research_drivers(question)
        assert len(result) <= 8
        assert all(isinstance(d, ScoredDriver) for d in result)

    @patch("forecasting_tools.agents_and_tools.research.drivers_researcher.GeneralLlm")
    async def test_low_viability_filters_candidates(
        self, mock_llm_cls: AsyncMock
    ) -> None:
        from code_tests.unit_tests.forecasting_test_manager import (
            ForecastingTestManager,
        )

        question = ForecastingTestManager.get_fake_binary_question()

        candidates = [_make_candidate(f"Driver {i}") for i in range(4)]
        # All candidates get low viability scores
        precondition_assessments = [
            PreconditionAssessment(
                index=i,
                dominance_plausibility="very_low",
                precondition_summary="Not met",
                viability_score=0.1,
            )
            for i in range(4)
        ]

        mock_llm_instance = AsyncMock()
        mock_llm_instance.invoke_and_return_verified_type = AsyncMock(
            side_effect=[candidates, precondition_assessments]
        )
        mock_llm_cls.return_value = mock_llm_instance

        result = await DriversResearcher.research_drivers(
            question, num_drivers_to_return=4
        )
        # All filtered out by low viability, _score_and_select gets empty list
        assert len(result) == 0

    @patch("forecasting_tools.agents_and_tools.research.drivers_researcher.GeneralLlm")
    async def test_empty_candidates_returns_empty(
        self, mock_llm_cls: AsyncMock
    ) -> None:
        from code_tests.unit_tests.forecasting_test_manager import (
            ForecastingTestManager,
        )

        question = ForecastingTestManager.get_fake_binary_question()

        mock_llm_instance = AsyncMock()
        mock_llm_instance.invoke_and_return_verified_type = AsyncMock(
            return_value=[]
        )
        mock_llm_cls.return_value = mock_llm_instance

        result = await DriversResearcher.research_drivers(question)
        assert result == []


class TestPreconditionModels:
    def test_precondition_status_enum(self) -> None:
        assert len(PreconditionStatus) == 4
        assert PreconditionStatus.EMERGING.value == "emerging"
        assert PreconditionStatus.STABLE.value == "stable"
        assert PreconditionStatus.ABSENT.value == "absent"
        assert PreconditionStatus.CONTRARY.value == "contrary"

    def test_precondition_model(self) -> None:
        precondition = Precondition(
            description="Test precondition",
            why_necessary="Because it's needed",
            status=PreconditionStatus.EMERGING,
            evidence_summary="Some evidence",
            citations=["[1](https://example.com)"],
        )
        assert precondition.description == "Test precondition"
        assert precondition.status == PreconditionStatus.EMERGING
        assert len(precondition.citations) == 1

    def test_precondition_optional_fields(self) -> None:
        precondition = Precondition(
            description="Test",
            why_necessary="Reason",
        )
        assert precondition.status is None
        assert precondition.evidence_summary is None
        assert precondition.citations == []

    def test_dominance_scenario_model(self) -> None:
        scenario = DominanceScenario(
            scenario_description="This driver dominates when X happens",
            timescale_plausibility="high",
            system_effects=["Effect 1", "Effect 2"],
        )
        assert scenario.timescale_plausibility == "high"
        assert len(scenario.system_effects) == 2

    def test_precondition_analysis_model(self) -> None:
        scenario = DominanceScenario(
            scenario_description="Test scenario",
            timescale_plausibility="medium",
            system_effects=["Effect 1"],
        )
        preconditions = [
            Precondition(
                description="Pre 1",
                why_necessary="Why 1",
                status=PreconditionStatus.EMERGING,
            ),
            Precondition(
                description="Pre 2",
                why_necessary="Why 2",
                status=PreconditionStatus.ABSENT,
            ),
        ]
        analysis = PreconditionAnalysis(
            driver_name="Test Driver",
            dominance_scenario=scenario,
            preconditions=preconditions,
            precondition_alignment_score=0.5,
            overall_emergence_strength="moderate",
        )
        assert analysis.driver_name == "Test Driver"
        assert analysis.precondition_alignment_score == 0.5
        assert len(analysis.preconditions) == 2

    def test_precondition_analysis_score_bounds(self) -> None:
        scenario = DominanceScenario(
            scenario_description="Test",
            timescale_plausibility="low",
            system_effects=[],
        )
        with pytest.raises(Exception):
            PreconditionAnalysis(
                driver_name="Bad",
                dominance_scenario=scenario,
                preconditions=[],
                precondition_alignment_score=1.5,
                overall_emergence_strength="weak",
            )


class TestLlmPreconditionValidation:
    @patch("forecasting_tools.agents_and_tools.research.drivers_researcher.GeneralLlm")
    async def test_llm_precondition_validation_filters_by_viability(
        self, mock_llm_cls: AsyncMock
    ) -> None:
        candidates = [_make_candidate(f"Driver {i}") for i in range(3)]

        assessments = [
            PreconditionAssessment(
                index=0,
                dominance_plausibility="high",
                precondition_summary="Met",
                viability_score=0.9,
            ),
            PreconditionAssessment(
                index=1,
                dominance_plausibility="low",
                precondition_summary="Not met",
                viability_score=0.1,
            ),
            PreconditionAssessment(
                index=2,
                dominance_plausibility="medium",
                precondition_summary="Partially",
                viability_score=0.5,
            ),
        ]

        mock_llm_instance = AsyncMock()
        mock_llm_instance.invoke_and_return_verified_type = AsyncMock(
            return_value=assessments
        )
        mock_llm_cls.return_value = mock_llm_instance

        validated = await DriversResearcher._llm_precondition_validation(
            "Test question", candidates
        )

        # Index 1 should be filtered (viability 0.1 < 0.3)
        assert len(validated) == 2
        assert validated[0].name == "Driver 0"
        assert validated[1].name == "Driver 2"

    async def test_llm_precondition_validation_empty_input(self) -> None:
        validated = await DriversResearcher._llm_precondition_validation(
            "Test question", []
        )
        assert validated == []
