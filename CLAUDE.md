# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python library for AI-powered probabilistic forecasting, primarily integrated with Metaculus.com. Uses LLMs and research tools to generate predictions on questions. Built with Poetry, Python 3.11+.

## Common Commands

```bash
# Install dependencies
poetry install

# Run all unit tests (parallel by default via pytest-xdist)
pytest ./code_tests/unit_tests

# Run a single test file
pytest ./code_tests/unit_tests/test_specific.py

# Run a single test
pytest ./code_tests/unit_tests/test_specific.py::test_name

# Run pre-commit checks (Black, isort, Ruff, typos)
pre-commit run --all-files

# Run Streamlit frontend
streamlit run front_end/main.py
```

## Architecture

### Core Abstractions

**ForecastBot** (`forecast_bots/forecast_bot.py`): Abstract base class for all forecasting bots. Override `run_research()` and `_run_forecast_on_binary/numeric/multiple_choice()`. `TemplateBot` is the default implementation. Official tournament bots live in `forecast_bots/official_bots/`.

**Question types** (`data_models/questions.py`): Pydantic models — `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`, `DateQuestion`, `DiscreteQuestion`, `ConditionalQuestion` — all inherit from `MetaculusQuestion`.

**Report types** (`data_models/forecast_report.py` and siblings): `BinaryReport`, `NumericReport`, etc. Each contains the question, explanation (markdown), prediction, cost estimate, and errors.

**GeneralLlm** (`ai_models/general_llm.py`): Unified LLM interface wrapping litellm (100+ models). Handles routing, retries, cost tracking, rate limiting, structured output via Pydantic.

### Data Flow

Question → Research (SmartSearcher, KeyFactorsResearcher, BaseRateResearcher) → Prediction(s) → Aggregation → ForecastReport → Metaculus API

### Key Packages

- `ai_models/` — LLM wrappers, Exa search, cost/rate management
- `agents_and_tools/` — Research tools (smart search, base rates, key factors, question decomposition, AI congress)
- `data_models/` — Pydantic question/report types, JSON serialization (`Jsonable` mixin)
- `helpers/` — Metaculus API client (`metaculus_client.py` is current, `metaculus_api.py` is deprecated), prediction extraction, AskNews integration
- `cp_benchmarking/` — Bot benchmarking framework
- `auto_optimizers/` — Automatic prompt optimization
- `util/` — Async batching, logging, stats, file utilities

### Entry Points

- `run_bots.py` — Main script for running bots in tournaments (GitHub Actions)
- `forecasting_tools/__init__.py` — Public API exports
- `front_end/main.py` — Streamlit web UI

## Code Conventions

- **Type hints required** on all function parameters and return types. Use modern syntax (`list[T]`, `dict[K,V]`), not `typing.List`/`typing.Dict`.
- **Black formatter** (88 char lines), **isort** (Black profile), **Ruff** linting.
- **Avoid comments** — prefer descriptive names. Only comment design decisions.
- **Async-first** — uses `asyncio` throughout with `nest_asyncio.apply()`. Batch concurrent work via `async_batching` utility.
- **Cost management** via context managers: `with MonetaryCostManager(max_cost) as mgr:`

## Testing Conventions

- `asyncio_mode = auto` in pytest.ini — do **not** use `@pytest.mark.asyncio`.
- When using `pytest.raises()`, do **not** assert on exception message text.
- Tests run in parallel (`-n auto` via pytest-xdist).
- Use `time_machine` for date mocking.

## Environment Variables

Copy `.env.template` to `.env`. Key variables: `OPENROUTER_API_KEY` (primary LLM), `METACULUS_TOKEN`, `ASKNEWS_CLIENT_ID`/`ASKNEWS_SECRET`, `EXA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Set `PYTHONPATH=.`.
