# Demand Planner

AI-powered supply chain demand forecasting system with interactive dashboard and GPT-4o assistant.

**[Live Dashboard](https://demand-planner-dashboard.vercel.app)** | **[API](https://demand-planner-production-b800.up.railway.app/api/skus)** | **[Documentation](./docs/)**

---

## Overview

Full-stack demand planning system that forecasts product demand using statistical models (SARIMA, Holt-Winters, Prophet), detects anomalies, and provides natural language analytics through a GPT-4o powered assistant.

Built with Python, FastAPI, React, LangChain, and OpenAI.

## Features

- **Multi-model forecasting** - SARIMA, Holt-Winters, Prophet, and weighted ensemble with configurable 7-90 day horizons
- **Anomaly detection** - Z-score based with adjustable sensitivity and spike/drop classification
- **Interactive dashboard** - Five analysis views: History, Forecast, Anomalies, Compare, Edit Data
- **AI assistant** - GPT-4o chat with tool-calling, RAG knowledge base, and automated reports
- **Data flexibility** - CSV upload, inline editing for scenario analysis, full reset capability
- **REST API** - 15 endpoints for forecasting, anomaly detection, data management, and LLM analytics

## Architecture

```
Frontend (React + Recharts)  -->  FastAPI Backend  -->  Forecasting Engine
     Vercel                        Railway              SARIMA / ETS / Prophet
                                      |
                                  LangChain Agent
                                      |
                                  GPT-4o (OpenAI)
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- OpenAI API key

### Backend

```bash
git clone https://github.com/Lakshya2905/demand-planner.git
cd demand-planner
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Start the server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
npx create-react-app demand-ui
cd demand-ui
npm install recharts
# Copy the dashboard component to src/App.js
npm start
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/skus` | List all SKUs |
| GET | `/api/history/{sku_id}` | Historical demand + stats |
| POST | `/api/forecast` | Run forecast (ARIMA/ETS/Prophet/ensemble) |
| POST | `/api/anomalies` | Detect demand anomalies |
| GET | `/api/compare?skus=A,B` | Compare multiple SKUs |
| POST | `/api/edit` | Edit a data point |
| POST | `/api/upload` | Upload new CSV dataset |
| POST | `/api/chat` | AI assistant (GPT-4o) |
| POST | `/api/forecast-chain` | LangChain chain: forecast + backtest + insights |
| POST | `/api/anomalies-chain` | LangChain chain: anomalies + explanation |
| POST | `/api/knowledge` | RAG knowledge base query |
| GET | `/api/usage` | LLM usage stats |

## Data Format

CSV with columns: `date`, `sku_id`, `demand`, `warehouse` (optional)

```csv
date,sku_id,demand,warehouse
2025-01-01,SKU-001,150.5,warehouse-east
2025-01-02,SKU-001,142.3,warehouse-east
```

## Project Structure

```
demand-planner/
  api.py                         # FastAPI backend (15 endpoints)
  main.py                        # CLI entry point
  config/settings.yaml           # All model parameters
  src/
    agents/demand_agent.py       # LangChain agent with 7 tools
    chains/
      forecast_chain.py          # 5-step RunnableSequence
      anomaly_chain.py           # Detection + LLM explanation
      rag_chain.py               # RAG knowledge base (10 docs)
      memory.py                  # Auto-summarizing chat memory
      callbacks.py               # Token/cost/latency tracking
      output_parsers.py          # Pydantic structured outputs
    models/
      arima_model.py             # SARIMA(1,1,1)(1,1,1,7)
      exponential_smoothing.py   # Holt-Winters
      prophet_model.py           # Meta Prophet
    tools/                       # LangChain tools
    data/                        # Preprocessor + validators
  sample_data/                   # Sample data generator
```

## LangChain Components

| Component | Purpose |
|-----------|---------|
| Demand Agent | Tool-calling agent (ReAct pattern) with 7 tools |
| Forecast Chain | RunnableSequence: validate, preprocess, fit, backtest, LLM insights |
| Anomaly Chain | Statistical detection + GPT-4o business explanation |
| RAG Knowledge | 10-document vector store for supply chain concepts |
| Memory | Conversation history with automatic summarization |
| Callbacks | Token tracking, cost estimation, latency logging |
| Output Parsers | Pydantic models for structured LLM responses |

## Tech Stack

Python 3.12, FastAPI, React 18, Recharts, LangChain, OpenAI GPT-4o, Statsmodels, Prophet, Pandas, NumPy

Hosted on Railway (backend) and Vercel (frontend).

## License

MIT
