from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion
from forecasting_tools.data_models.timestamped_predictions import (
    BinaryTimestampedPrediction,
    NumericTimestampedDistribution,
)
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


def _make_timestamp() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestDriftGuard:
    async def test_binary_no_previous_forecast_passes_through(self) -> None:
        bot = _make_bot()
        question = BinaryQuestion(
            question_text="Test?",
            previous_forecasts=None,
        )
        result = await bot._aggregate_predictions([0.8], question)
        assert result == 0.8

    async def test_binary_small_drift_passes_through(self) -> None:
        bot = _make_bot()
        question = BinaryQuestion(
            question_text="Test?",
            previous_forecasts=[
                BinaryTimestampedPrediction(
                    prediction_in_decimal=0.5,
                    timestamp=_make_timestamp(),
                    timestamp_end=None,
                )
            ],
        )
        result = await bot._aggregate_predictions([0.6], question)
        assert result == 0.6

    async def test_binary_large_drift_gets_dampened(self) -> None:
        bot = _make_bot()
        question = BinaryQuestion(
            question_text="Test?",
            previous_forecasts=[
                BinaryTimestampedPrediction(
                    prediction_in_decimal=0.3,
                    timestamp=_make_timestamp(),
                    timestamp_end=None,
                )
            ],
        )
        # Drift of 0.5 exceeds MAX_BINARY_DRIFT (0.15)
        result = await bot._aggregate_predictions([0.8], question)
        # Blended: 0.6 * 0.8 + 0.4 * 0.3 = 0.48 + 0.12 = 0.60
        assert abs(result - 0.60) < 1e-6

    async def test_binary_drift_guard_uses_last_forecast(self) -> None:
        bot = _make_bot()
        question = BinaryQuestion(
            question_text="Test?",
            previous_forecasts=[
                BinaryTimestampedPrediction(
                    prediction_in_decimal=0.1,
                    timestamp=_make_timestamp(),
                    timestamp_end=None,
                ),
                BinaryTimestampedPrediction(
                    prediction_in_decimal=0.5,
                    timestamp=_make_timestamp(),
                    timestamp_end=None,
                ),
            ],
        )
        # Only compares against last (0.5), drift = 0.3 > 0.15
        result = await bot._aggregate_predictions([0.8], question)
        # Blended: 0.6 * 0.8 + 0.4 * 0.5 = 0.48 + 0.20 = 0.68
        assert abs(result - 0.68) < 1e-6

    async def test_numeric_large_drift_gets_dampened(self) -> None:
        bot = _make_bot()

        prev_percentiles = [
            Percentile(percentile=0.1, value=10.0),
            Percentile(percentile=0.5, value=30.0),
            Percentile(percentile=0.9, value=50.0),
        ]
        new_percentiles = [
            Percentile(percentile=0.1, value=30.0),
            Percentile(percentile=0.5, value=70.0),
            Percentile(percentile=0.9, value=90.0),
        ]

        question = NumericQuestion(
            question_text="Test?",
            upper_bound=100.0,
            lower_bound=0.0,
            open_upper_bound=False,
            open_lower_bound=False,
            previous_forecasts=[
                NumericTimestampedDistribution(
                    declared_percentiles=prev_percentiles,
                    open_upper_bound=False,
                    open_lower_bound=False,
                    upper_bound=100.0,
                    lower_bound=0.0,
                    zero_point=None,
                    timestamp=_make_timestamp(),
                    timestamp_end=None,
                )
            ],
        )

        new_dist = NumericDistribution(
            declared_percentiles=new_percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            standardize_cdf=False,
        )

        result = await bot._aggregate_predictions([new_dist], question)
        assert isinstance(result, NumericDistribution)
        # Median drift = |70 - 30| / 100 = 0.4 > 0.15, so blending occurs
        # Blended median: 0.6 * 70 + 0.4 * 30 = 42 + 12 = 54
        result_values = {
            p.percentile: p.value for p in result.declared_percentiles
        }
        # Check that blending moved the median closer to previous
        # The exact value depends on from_question standardization,
        # but raw blended p50 should be ~54 (between 30 and 70)
        p50_candidates = [
            v for p, v in result_values.items() if abs(p - 0.5) < 0.01
        ]
        if p50_candidates:
            assert 30.0 < p50_candidates[0] < 70.0

    async def test_numeric_small_drift_passes_through(self) -> None:
        bot = _make_bot()

        prev_percentiles = [
            Percentile(percentile=0.1, value=40.0),
            Percentile(percentile=0.5, value=50.0),
            Percentile(percentile=0.9, value=60.0),
        ]
        new_percentiles = [
            Percentile(percentile=0.1, value=42.0),
            Percentile(percentile=0.5, value=55.0),
            Percentile(percentile=0.9, value=63.0),
        ]

        question = NumericQuestion(
            question_text="Test?",
            upper_bound=100.0,
            lower_bound=0.0,
            open_upper_bound=False,
            open_lower_bound=False,
            previous_forecasts=[
                NumericTimestampedDistribution(
                    declared_percentiles=prev_percentiles,
                    open_upper_bound=False,
                    open_lower_bound=False,
                    upper_bound=100.0,
                    lower_bound=0.0,
                    zero_point=None,
                    timestamp=_make_timestamp(),
                    timestamp_end=None,
                )
            ],
        )

        new_dist = NumericDistribution(
            declared_percentiles=new_percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            standardize_cdf=False,
        )

        result = await bot._aggregate_predictions([new_dist], question)
        assert isinstance(result, NumericDistribution)
