# Demand Planning & Forecasting System

An AI-powered supply chain demand planning and forecasting platform built with **Claude Code**, **LangChain**, and **Anthropic's Claude API**.

## Features

- **Time Series Forecasting**: ARIMA, Exponential Smoothing, and Prophet-based models for inventory demand prediction
- **Anomaly Detection**: Automated identification of unusual demand spikes, drops, and seasonal deviations
- **Natural Language Querying**: Ask questions about your data in plain English (e.g., "What's the projected demand for SKU-1234 next month?")
- **Automated Report Generation**: Weekly/monthly demand reports with insights, trends, and recommendations

## Architecture

```
demand-planner/
├── src/
│   ├── agents/              # LangChain agents for orchestration
│   │   ├── demand_agent.py  # Main demand planning agent
│   │   └── report_agent.py  # Report generation agent
│   ├── chains/              # LangChain chains
│   │   ├── forecast_chain.py
│   │   ├── anomaly_chain.py
│   │   └── query_chain.py
│   ├── tools/               # Custom LangChain tools
│   │   ├── forecasting.py
│   │   ├── anomaly_detector.py
│   │   ├── data_loader.py
│   │   └── report_generator.py
│   ├── models/              # Forecasting models
│   │   ├── arima_model.py
│   │   ├── exponential_smoothing.py
│   │   └── prophet_model.py
│   ├── data/                # Data processing
│   │   ├── preprocessor.py
│   │   └── validators.py
│   └── utils/               # Utilities
│       ├── config.py
│       └── logger.py
├── config/
│   └── settings.yaml        # Configuration file
├── tests/                   # Unit and integration tests
├── sample_data/             # Sample CSV data for testing
├── docs/                    # Documentation
├── main.py                  # Entry point
├── requirements.txt
├── pyproject.toml
├── .env.example
└── CLAUDE.md                # Claude Code project instructions
```

## Prerequisites

- Python 3.10+
- Node.js 18+ (for Claude Code)
- An Anthropic API key

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd demand-planner
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run with sample data

```bash
python main.py --demo
```

### 4. Interactive mode (Natural Language Querying)

```bash
python main.py --interactive
```

### 5. Generate a forecast report

```bash
python main.py --report --sku "SKU-1234" --horizon 30
```

## Using with Claude Code

This project includes a `CLAUDE.md` file that gives Claude Code full context about the codebase. You can use Claude Code to:

```bash
# Install Claude Code (requires Node.js 18+)
npm install -g @anthropic-ai/claude-code

# Navigate to project directory and start Claude Code
cd demand-planner
claude

# Example prompts in Claude Code:
# "Add a new forecasting model using XGBoost"
# "Write tests for the anomaly detection chain"
# "Optimize the data preprocessing pipeline for large datasets"
```

## Configuration

Edit `config/settings.yaml` to customize:

- Forecasting horizons and model parameters
- Anomaly detection thresholds
- Report templates and scheduling
- Data source connections

## Sample Queries (Interactive Mode)

```
> What is the forecasted demand for SKU-1234 over the next 4 weeks?
> Show me anomalies detected in warehouse-east inventory last month
> Compare actual vs predicted demand for Q3
> Generate a weekly demand summary report
> Which SKUs are trending upward this quarter?
```

## License

MIT
