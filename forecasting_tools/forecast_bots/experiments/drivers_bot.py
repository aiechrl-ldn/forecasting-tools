import asyncio
import logging

from forecasting_tools.agents_and_tools.research.base_rate_researcher import (
    BaseRateEstimate,
    LightweightBaseRateResearcher,
)
from forecasting_tools.agents_and_tools.research.drivers_researcher import (
    DriversResearcher,
    ScoredDriver,
)
from forecasting_tools.agents_and_tools.research.key_factors_researcher import (
    KeyFactorsResearcher,
    ScoredKeyFactor,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_spring import (
    SpringTemplateBot2026,
)
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher

logger = logging.getLogger(__name__)

MAX_BINARY_DRIFT = 0.15
MAX_NUMERIC_DRIFT = 0.15
NEW_WEIGHT = 0.6


class DriversBot(SpringTemplateBot2026):

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": GeneralLlm(
                model="openrouter/anthropic/claude-opus-4.6",
                temperature=0.3,
            ),
            "summarizer": GeneralLlm(
                model="openrouter/anthropic/claude-sonnet-4.5",
                temperature=0.3,
            ),
            "researcher": "asknews/news-summaries",
            "parser": GeneralLlm(
                model="openrouter/anthropic/claude-sonnet-4.5",
                temperature=0.3,
            ),
        }

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        aggregate = await super()._aggregate_predictions(predictions, question)

        if not question.previous_forecasts:
            return aggregate

        if isinstance(question, BinaryQuestion):
            assert isinstance(aggregate, float)
            previous = question.previous_forecasts[-1].prediction_in_decimal
            drift = abs(aggregate - previous)
            if drift > MAX_BINARY_DRIFT:
                aggregate = NEW_WEIGHT * aggregate + (1 - NEW_WEIGHT) * previous
                aggregate = max(0.001, min(0.999, aggregate))
                logger.info(
                    f"Drift guard: binary drift {drift:.3f} exceeded "
                    f"{MAX_BINARY_DRIFT}, blended to {aggregate:.3f}"
                )

        elif isinstance(question, NumericQuestion):
            assert isinstance(aggregate, NumericDistribution)
            prev_dist = question.previous_forecasts[-1]
            prev_percentiles = {
                p.percentile: p.value
                for p in prev_dist.declared_percentiles
            }
            new_percentiles = {
                p.percentile: p.value
                for p in aggregate.declared_percentiles
            }

            # Compare medians to decide if dampening is needed
            common_points = sorted(
                set(prev_percentiles.keys()) & set(new_percentiles.keys())
            )
            if common_points:
                mid_idx = len(common_points) // 2
                mid_pct = common_points[mid_idx]
                prev_median = prev_percentiles[mid_pct]
                new_median = new_percentiles[mid_pct]
                range_span = question.upper_bound - question.lower_bound
                if range_span > 0:
                    relative_drift = abs(new_median - prev_median) / range_span
                    if relative_drift > MAX_NUMERIC_DRIFT:
                        blended = [
                            Percentile(
                                percentile=p.percentile,
                                value=(
                                    NEW_WEIGHT * p.value
                                    + (1 - NEW_WEIGHT)
                                    * prev_percentiles.get(
                                        p.percentile, p.value
                                    )
                                ),
                            )
                            for p in aggregate.declared_percentiles
                        ]
                        aggregate = NumericDistribution(
                            declared_percentiles=blended,
                            open_upper_bound=aggregate.open_upper_bound,
                            open_lower_bound=aggregate.open_lower_bound,
                            upper_bound=aggregate.upper_bound,
                            lower_bound=aggregate.lower_bound,
                            zero_point=aggregate.zero_point,
                            cdf_size=aggregate.cdf_size,
                            standardize_cdf=False,
                            is_date=aggregate.is_date,
                        )
                        aggregate = NumericDistribution.from_question(
                            aggregate.declared_percentiles, question
                        )
                        logger.info(
                            f"Drift guard: numeric drift {relative_drift:.3f} "
                            f"exceeded {MAX_NUMERIC_DRIFT}, blended CDF"
                        )

        return aggregate

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Streams A (LLM-only), B (AskNews), D (LLM-only) in parallel
            drivers_task = DriversResearcher.research_drivers(question)
            key_factors_task = KeyFactorsResearcher.find_and_sort_key_factors(
                question,
                num_key_factors_to_return=5,
                num_questions_to_research_with=10,
            )
            base_rates_task = LightweightBaseRateResearcher.research_base_rates(question)

            results = await asyncio.gather(
                drivers_task,
                key_factors_task,
                base_rates_task,
                return_exceptions=True,
            )

            # Process Stream A results
            drivers_section = ""
            if isinstance(results[0], BaseException):
                logger.exception(
                    "Driver research failed, continuing without",
                    exc_info=results[0],
                )
            else:
                validated_drivers: list[ScoredDriver] = results[0]
                drivers_section = (
                    "## STEEP Driver Analysis\n"
                    + ScoredDriver.turn_drivers_into_markdown(validated_drivers)
                    + "\n\n"
                )

            # Process Stream B results
            key_factors_section = ""
            if isinstance(results[1], BaseException):
                logger.exception(
                    "Key factors research failed, continuing without",
                    exc_info=results[1],
                )
            else:
                key_factors: list[ScoredKeyFactor] = results[1]
                if key_factors:
                    key_factors_section = (
                        "## Key Factors\n"
                        + ScoredKeyFactor.turn_key_factors_into_markdown_list(
                            key_factors
                        )
                        + "\n\n"
                    )

            # Process Stream D results (base rates)
            base_rates_section = ""
            if isinstance(results[2], BaseException):
                logger.exception(
                    "Base rate research failed, continuing without",
                    exc_info=results[2],
                )
            else:
                base_rates: list[BaseRateEstimate] = results[2]
                if base_rates:
                    base_rates_section = (
                        "## Base Rate Analysis\n"
                        + BaseRateEstimate.format_as_markdown(base_rates)
                        + "\n\n"
                    )

            # Stream C: Latest News
            news_section = ""
            try:
                news_section = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            except Exception as e:
                logger.warning(f"AskNews research failed: {e}", exc_info=True)

            research = (
                drivers_section
                + key_factors_section
                + base_rates_section
                + news_section
            )
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research
