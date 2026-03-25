# CLAUDE.md - Claude Code Project Context

## Project Overview
This is a demand planning and forecasting system for supply chain/inventory management.
It uses LangChain with Claude (Anthropic) as the LLM backbone.

## Tech Stack
- Python 3.10+
- LangChain + LangChain-OpenAI for agent orchestration
- OpenAI GPT-4o as the default model
- Statsmodels (ARIMA, Exponential Smoothing) for statistical forecasting
- Prophet for time series forecasting
- Pandas/NumPy for data manipulation
- Scipy for anomaly detection

## Project Structure
- `src/agents/` - LangChain agents that orchestrate tools and chains
- `src/chains/` - LangChain chains for specific workflows (forecast, anomaly, query)
- `src/tools/` - Custom LangChain tools wrapping forecasting and analysis logic
- `src/models/` - Pure forecasting model implementations (no LangChain dependency)
- `src/data/` - Data loading, preprocessing, validation
- `src/utils/` - Configuration, logging, shared utilities
- `config/settings.yaml` - All tunable parameters
- `tests/` - Pytest-based tests

## Key Design Decisions
- Models in `src/models/` are pure Python with no LangChain coupling, making them testable independently
- LangChain tools in `src/tools/` wrap model logic and expose it to the agent
- The demand agent uses a ReAct pattern to decide which tools to invoke
- All configuration is centralized in `config/settings.yaml`

## Running
- `python main.py --demo` runs with sample data
- `python main.py --interactive` starts the NL query interface
- `python main.py --report --sku SKU-ID --horizon N` generates a forecast report
- `pytest tests/` runs the test suite

## Conventions
- Type hints on all function signatures
- Docstrings on all public functions
- Logging via `src/utils/logger.py` (not print statements)
- Config access via `src/utils/config.py` (not hardcoded values)
