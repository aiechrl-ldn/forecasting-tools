from unittest.mock import AsyncMock, patch

from forecasting_tools.agents_and_tools.research.base_rate_researcher import (
    BaseRateEstimate,
    LightweightBaseRateResearcher,
)

LLM_PATCH = (
    "forecasting_tools.agents_and_tools.research."
    "base_rate_researcher.GeneralLlm"
)


def _make_estimate(
    reference_class: str = "Test reference class",
    numerator: int = 3,
    denominator: int = 10,
) -> BaseRateEstimate:
    return BaseRateEstimate(
        reference_class=reference_class,
        numerator_description="Events matching criteria",
        denominator_description="Total events",
        numerator=numerator,
        denominator=denominator,
        historical_rate=numerator / denominator,
        time_period="2000-2025",
        relevance_reasoning="Relevant because it matches the question scope",
    )


class TestBaseRateEstimate:
    def test_format_as_markdown_with_estimates(self) -> None:
        estimates = [
            _make_estimate("US shutdowns", 4, 20),
            _make_estimate("Budget crises", 6, 30),
        ]
        result = BaseRateEstimate.format_as_markdown(estimates)
        assert "**US shutdowns**" in result
        assert "4/20" in result
        assert "**Budget crises**" in result
        assert "6/30" in result

    def test_format_as_markdown_empty(self) -> None:
        result = BaseRateEstimate.format_as_markdown([])
        assert result == "No base rates identified."

    def test_format_as_markdown_rate_percentage(self) -> None:
        estimates = [_make_estimate(numerator=1, denominator=4)]
        result = BaseRateEstimate.format_as_markdown(estimates)
        assert "25%" in result


class TestLightweightBaseRateResearcher:
    @patch(LLM_PATCH)
    async def test_research_base_rates_returns_estimates(
        self,
        mock_llm_cls: AsyncMock,
    ) -> None:
        from code_tests.unit_tests.forecasting_test_manager import (
            ForecastingTestManager,
        )

        expected = [_make_estimate(), _make_estimate("Another class", 5, 20)]

        mock_instance = AsyncMock()
        mock_instance.invoke_and_return_verified_type = AsyncMock(
            return_value=expected
        )
        mock_llm_cls.return_value = mock_instance

        question = ForecastingTestManager.get_fake_binary_question()
        result = await LightweightBaseRateResearcher.research_base_rates(question)

        assert len(result) == 2
        assert result[0].reference_class == "Test reference class"
        mock_instance.invoke_and_return_verified_type.assert_called_once()

    @patch(LLM_PATCH)
    async def test_research_base_rates_custom_count(
        self,
        mock_llm_cls: AsyncMock,
    ) -> None:
        from code_tests.unit_tests.forecasting_test_manager import (
            ForecastingTestManager,
        )

        mock_instance = AsyncMock()
        mock_instance.invoke_and_return_verified_type = AsyncMock(
            return_value=[_make_estimate()]
        )
        mock_llm_cls.return_value = mock_instance

        question = ForecastingTestManager.get_fake_binary_question()
        await LightweightBaseRateResearcher.research_base_rates(
            question, num_base_rates=1
        )

        call_args = mock_instance.invoke_and_return_verified_type.call_args
        prompt = call_args[0][0]
        assert "1" in prompt
