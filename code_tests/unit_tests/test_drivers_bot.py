from unittest.mock import AsyncMock, MagicMock, patch

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.forecast_bots.experiments.drivers_bot import DriversBot

ASKNEWS_PATCH = (
    "forecasting_tools.forecast_bots.experiments.drivers_bot.AskNewsSearcher"
)


def _make_bot() -> DriversBot:
    return DriversBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
    )


def _mock_asknews() -> MagicMock:
    mock_instance = AsyncMock()
    mock_instance.get_formatted_news_async = AsyncMock(return_value="AskNews results")
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


class TestDriversBot:
    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.BaseRateResearcher")
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.DriversResearcher")
    async def test_run_research_includes_steep_section(
        self,
        mock_researcher: AsyncMock,
        mock_base_rate: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        mock_researcher.research_drivers = AsyncMock(
            return_value=[_make_scored_driver("AI Progress")]
        )

        # Mock base rate to raise to skip it
        mock_base_rate.side_effect = ValueError("Skip base rate")

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## STEEP Driver Analysis" in result
        assert "AI Progress" in result

    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.BaseRateResearcher")
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.DriversResearcher")
    async def test_run_research_fallback_on_failure(
        self,
        mock_researcher: AsyncMock,
        mock_base_rate: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        mock_researcher.research_drivers = AsyncMock(
            side_effect=RuntimeError("Driver research failed")
        )
        mock_base_rate.side_effect = ValueError("Skip base rate")

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## STEEP Driver Analysis" not in result


class TestDriversBotBaseRate:
    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.DriversResearcher")
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.BaseRateResearcher")
    async def test_base_rate_included_when_successful(
        self,
        mock_base_rate_cls: MagicMock,
        mock_researcher: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        # Mock base rate report
        mock_report = MagicMock()
        mock_report.markdown_report = "Historical rate: 10% per year"

        mock_instance = AsyncMock()
        mock_instance.make_base_rate_report = AsyncMock(return_value=mock_report)
        mock_base_rate_cls.return_value = mock_instance

        mock_researcher.research_drivers = AsyncMock(
            return_value=[_make_scored_driver("Driver 1")]
        )

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## Historical Base Rate" in result
        assert "10% per year" in result

    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.DriversResearcher")
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.BaseRateResearcher")
    async def test_base_rate_context_passed_to_drivers(
        self,
        mock_base_rate_cls: MagicMock,
        mock_researcher: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        mock_report = MagicMock()
        mock_report.markdown_report = "Historical rate"

        mock_instance = AsyncMock()
        mock_instance.make_base_rate_report = AsyncMock(return_value=mock_report)
        mock_base_rate_cls.return_value = mock_instance

        mock_researcher.research_drivers = AsyncMock(return_value=[])

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        await bot.run_research(question)

        # Verify base_rate_context was passed
        call_kwargs = mock_researcher.research_drivers.call_args.kwargs
        assert "base_rate_context" in call_kwargs
        assert call_kwargs["base_rate_context"] == mock_report


class TestDriverGuidedSearch:
    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.SmartSearcher")
    async def test_driver_guided_search_returns_results(
        self, mock_searcher_cls: MagicMock
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        mock_searcher_instance = AsyncMock()
        mock_searcher_instance.invoke = AsyncMock(
            return_value="Deep dive research on driver"
        )
        mock_searcher_cls.return_value = mock_searcher_instance

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()
        drivers = [_make_scored_driver("Test Driver")]

        result = await bot._driver_guided_search(question, drivers)

        assert "## Driver Deep-Dive Research" in result
        assert "### Test Driver" in result
        assert "Deep dive research on driver" in result

    async def test_driver_guided_search_empty_drivers(self) -> None:
        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot._driver_guided_search(question, [])

        assert result == ""

    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.SmartSearcher")
    async def test_driver_guided_search_limits_to_five(
        self, mock_searcher_cls: MagicMock
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        mock_searcher_instance = AsyncMock()
        mock_searcher_instance.invoke = AsyncMock(return_value="Research")
        mock_searcher_cls.return_value = mock_searcher_instance

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()
        drivers = [_make_scored_driver(f"Driver {i}") for i in range(10)]

        await bot._driver_guided_search(question, drivers)

        # Should only call invoke 5 times (limit)
        assert mock_searcher_instance.invoke.call_count == 5

    @patch("forecasting_tools.forecast_bots.experiments.drivers_bot.SmartSearcher")
    async def test_driver_guided_search_handles_failures(
        self, mock_searcher_cls: MagicMock
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        mock_searcher_instance = AsyncMock()
        mock_searcher_instance.invoke = AsyncMock(
            side_effect=RuntimeError("Search failed")
        )
        mock_searcher_cls.return_value = mock_searcher_instance

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()
        drivers = [_make_scored_driver("Test Driver")]

        # Should not raise, just return empty
        result = await bot._driver_guided_search(question, drivers)

        assert result == ""
