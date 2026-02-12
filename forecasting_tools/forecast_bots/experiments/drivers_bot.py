import asyncio
import logging

from forecasting_tools.agents_and_tools.research.drivers_researcher import (
    DriversResearcher,
    ScoredDriver,
)
from forecasting_tools.agents_and_tools.research.key_factors_researcher import (
    KeyFactorsResearcher,
    ScoredKeyFactor,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_spring import (
    SpringTemplateBot2026,
)
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher

logger = logging.getLogger(__name__)


class DriversBot(SpringTemplateBot2026):

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": GeneralLlm(
                model="openrouter/anthropic/claude-opus-4.6",
                temperature=1,
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

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Stream A (LLM-only) and Stream B (AskNews) run in parallel
            drivers_task = DriversResearcher.research_drivers(question)
            key_factors_task = KeyFactorsResearcher.find_and_sort_key_factors(
                question,
                num_key_factors_to_return=5,
                num_questions_to_research_with=10,
            )

            results = await asyncio.gather(
                drivers_task, key_factors_task, return_exceptions=True
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

            # Stream C: Latest News
            news_section = ""
            try:
                news_section = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            except Exception as e:
                logger.warning(f"AskNews research failed: {e}", exc_info=True)

            research = drivers_section + key_factors_section + news_section
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research
