from unittest.mock import AsyncMock, MagicMock, patch

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.forecast_bots.experiments.drivers_bot import DriversBot

ASKNEWS_PATCH = (
    "forecasting_tools.forecast_bots.experiments.drivers_bot.AskNewsSearcher"
)
BASE_RATES_PATCH = (
    "forecasting_tools.forecast_bots.experiments.drivers_bot.LightweightBaseRateResearcher"
)
DRIVERS_PATCH = (
    "forecasting_tools.forecast_bots.experiments.drivers_bot.DriversResearcher"
)
KEY_FACTORS_PATCH = (
    "forecasting_tools.forecast_bots.experiments.drivers_bot.KeyFactorsResearcher"
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


def _mock_key_factors() -> list[MagicMock]:
    mock_factor = MagicMock()
    mock_factor.display_text = "Key factor 1 [Source](https://example.com)"
    return [mock_factor]


class TestDriversBot:
    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch(BASE_RATES_PATCH)
    @patch(KEY_FACTORS_PATCH)
    @patch(DRIVERS_PATCH)
    async def test_run_research_includes_steep_section(
        self,
        mock_researcher: AsyncMock,
        mock_key_factors: AsyncMock,
        mock_base_rates: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        mock_researcher.research_drivers = AsyncMock(
            return_value=[_make_scored_driver("AI Progress")]
        )
        mock_key_factors.find_and_sort_key_factors = AsyncMock(return_value=[])
        mock_base_rates.research_base_rates = AsyncMock(return_value=[])

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## STEEP Driver Analysis" in result
        assert "AI Progress" in result

    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch(BASE_RATES_PATCH)
    @patch(KEY_FACTORS_PATCH)
    @patch(DRIVERS_PATCH)
    async def test_run_research_includes_key_factors(
        self,
        mock_researcher: AsyncMock,
        mock_key_factors_cls: AsyncMock,
        mock_base_rates: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        mock_researcher.research_drivers = AsyncMock(
            return_value=[_make_scored_driver()]
        )

        mock_factor = MagicMock()
        mock_factor.display_text = "Important factor [Source](https://example.com)"
        mock_key_factors_cls.find_and_sort_key_factors = AsyncMock(
            return_value=[mock_factor]
        )
        mock_base_rates.research_base_rates = AsyncMock(return_value=[])

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## Key Factors" in result

    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch(BASE_RATES_PATCH)
    @patch(KEY_FACTORS_PATCH)
    @patch(DRIVERS_PATCH)
    async def test_run_research_includes_base_rates(
        self,
        mock_researcher: AsyncMock,
        mock_key_factors: AsyncMock,
        mock_base_rates: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        from code_tests.unit_tests.test_base_rate_researcher import _make_estimate

        mock_researcher.research_drivers = AsyncMock(return_value=[])
        mock_key_factors.find_and_sort_key_factors = AsyncMock(return_value=[])
        mock_base_rates.research_base_rates = AsyncMock(
            return_value=[_make_estimate("Gov shutdowns", 4, 20)]
        )

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## Base Rate Analysis" in result
        assert "Gov shutdowns" in result

    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch(BASE_RATES_PATCH)
    @patch(KEY_FACTORS_PATCH)
    @patch(DRIVERS_PATCH)
    async def test_run_research_fallback_on_driver_failure(
        self,
        mock_researcher: AsyncMock,
        mock_key_factors: AsyncMock,
        mock_base_rates: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        mock_researcher.research_drivers = AsyncMock(
            side_effect=RuntimeError("Driver research failed")
        )
        mock_key_factors.find_and_sort_key_factors = AsyncMock(return_value=[])
        mock_base_rates.research_base_rates = AsyncMock(return_value=[])

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## STEEP Driver Analysis" not in result

    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch(BASE_RATES_PATCH)
    @patch(KEY_FACTORS_PATCH)
    @patch(DRIVERS_PATCH)
    async def test_run_research_fallback_on_key_factors_failure(
        self,
        mock_researcher: AsyncMock,
        mock_key_factors: AsyncMock,
        mock_base_rates: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        mock_researcher.research_drivers = AsyncMock(
            return_value=[_make_scored_driver()]
        )
        mock_key_factors.find_and_sort_key_factors = AsyncMock(
            side_effect=RuntimeError("Key factors failed")
        )
        mock_base_rates.research_base_rates = AsyncMock(return_value=[])

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## STEEP Driver Analysis" in result
        assert "## Key Factors" not in result

    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch(BASE_RATES_PATCH)
    @patch(KEY_FACTORS_PATCH)
    @patch(DRIVERS_PATCH)
    async def test_run_research_fallback_on_base_rates_failure(
        self,
        mock_researcher: AsyncMock,
        mock_key_factors: AsyncMock,
        mock_base_rates: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        from code_tests.unit_tests.test_drivers_researcher import _make_scored_driver

        mock_researcher.research_drivers = AsyncMock(
            return_value=[_make_scored_driver()]
        )
        mock_key_factors.find_and_sort_key_factors = AsyncMock(return_value=[])
        mock_base_rates.research_base_rates = AsyncMock(
            side_effect=RuntimeError("Base rates failed")
        )

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "## STEEP Driver Analysis" in result
        assert "## Base Rate Analysis" not in result

    @patch(BASE_RATES_PATCH)
    @patch(KEY_FACTORS_PATCH)
    @patch(DRIVERS_PATCH)
    async def test_run_research_includes_news(
        self,
        mock_researcher: AsyncMock,
        mock_key_factors: AsyncMock,
        mock_base_rates: AsyncMock,
    ) -> None:
        mock_researcher.research_drivers = AsyncMock(return_value=[])
        mock_key_factors.find_and_sort_key_factors = AsyncMock(return_value=[])
        mock_base_rates.research_base_rates = AsyncMock(return_value=[])

        mock_asknews_instance = AsyncMock()
        mock_asknews_instance.get_formatted_news_async = AsyncMock(
            return_value="Latest news content"
        )

        with patch(ASKNEWS_PATCH) as mock_asknews_cls:
            mock_asknews_cls.return_value = mock_asknews_instance

            bot = _make_bot()
            question = ForecastingTestManager.get_fake_binary_question()

            result = await bot.run_research(question)

            assert "Latest news content" in result

    @patch(ASKNEWS_PATCH, new_callable=lambda: _mock_asknews)
    @patch(BASE_RATES_PATCH)
    @patch(KEY_FACTORS_PATCH)
    @patch(DRIVERS_PATCH)
    async def test_run_research_all_parallel_streams_fail_still_returns_news(
        self,
        mock_researcher: AsyncMock,
        mock_key_factors: AsyncMock,
        mock_base_rates: AsyncMock,
        mock_asknews: MagicMock,
    ) -> None:
        mock_researcher.research_drivers = AsyncMock(
            side_effect=RuntimeError("Drivers failed")
        )
        mock_key_factors.find_and_sort_key_factors = AsyncMock(
            side_effect=RuntimeError("Key factors failed")
        )
        mock_base_rates.research_base_rates = AsyncMock(
            side_effect=RuntimeError("Base rates failed")
        )

        bot = _make_bot()
        question = ForecastingTestManager.get_fake_binary_question()

        result = await bot.run_research(question)

        assert "AskNews results" in result
        assert "## STEEP Driver Analysis" not in result
        assert "## Key Factors" not in result
        assert "## Base Rate Analysis" not in result
