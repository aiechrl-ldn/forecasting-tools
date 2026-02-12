import logging

from forecasting_tools.agents_and_tools.research.drivers_researcher import (
    DriversResearcher,
    ScoredDriver,
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
            # 1. BASE RATE FIRST
            base_rate_section = ""
            base_rate_context = None
            try:
                from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
                    BaseRateResearcher,
                )

                base_rate_researcher = BaseRateResearcher(question.question_text)
                report = await base_rate_researcher.make_base_rate_report()
                base_rate_section = (
                    f"## Historical Base Rate\n{report.markdown_report}\n\n"
                )
                base_rate_context = report
            except Exception as e:
                logger.warning(f"Base rate research failed: {e}")

            # 2. DRIVERS (with base rate context)
            drivers_section = ""
            validated_drivers: list[ScoredDriver] = []
            try:
                validated_drivers = await DriversResearcher.research_drivers(
                    question,
                    base_rate_context=base_rate_context,
                )
                drivers_section = (
                    "## STEEP Driver Analysis\n"
                    + ScoredDriver.turn_drivers_into_markdown(validated_drivers)
                    + "\n\n"
                )
            except Exception:
                logger.exception("Driver research failed, continuing without")

            # 3. DRIVER-GUIDED SEARCH
            driver_research = await self._driver_guided_search(
                question, validated_drivers
            )

            # 4. ASKNEWS RESEARCH
            asknews_research = ""
            try:
                asknews_research = (
                    await AskNewsSearcher().get_formatted_news_async(
                        question.question_text
                    )
                )
            except Exception as e:
                logger.warning(f"AskNews research failed: {e}", exc_info=True)

            logger.info(
                f"Found Research for URL {question.page_url}:\n{asknews_research}"
            )
            return (
                base_rate_section + drivers_section + driver_research + asknews_research
            )

    async def _driver_guided_search(
        self,
        question: MetaculusQuestion,
        drivers: list[ScoredDriver],
    ) -> str:
        """Generate targeted searches based on validated drivers."""
        if not drivers:
            return ""

        async def _search_one_driver(driver: ScoredDriver) -> str | None:
            prompt = f"""
Analyze the following news context about a driver for a forecasting question.
Summarize the key findings relevant to this driver's trajectory and impact.

Question: {question.question_text}

Driver: {driver.name}
Mechanism: {driver.mechanism}
Direction of pressure: {driver.direction_of_pressure}

Focus on:
- Recent news/data on this driver's current trajectory
- Quantitative indicators if available
- Expert opinions on this driver's likely impact
"""
            try:
                news = await AskNewsSearcher().get_formatted_news_async(
                    f"{driver.name} {question.question_text}"
                )
                llm = GeneralLlm(
                    model="openrouter/anthropic/claude-sonnet-4.5",
                    temperature=0.3,
                )
                result = await llm.invoke(
                    f"{prompt}\n\nNews context:\n{news}"
                )
                return f"### {driver.name}\n{result}"
            except Exception:
                logger.warning(f"Driver search failed for {driver.name}")
                return None

        from forecasting_tools.util import async_batching

        coroutines = [_search_one_driver(d) for d in drivers[:5]]
        results, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                coroutines
            )
        )
        search_results = [r for r in results if r is not None]

        if search_results:
            return (
                "## Driver Deep-Dive Research\n" + "\n\n".join(search_results) + "\n\n"
            )
        return ""
