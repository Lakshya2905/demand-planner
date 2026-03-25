"""
FastAPI backend for Demand Planning Dashboard.
Exposes forecasting, anomaly detection, and data querying as REST endpoints.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import DemandDataPreprocessor
from src.models import ARIMAModel, ExponentialSmoothingModel, ProphetModel
from src.utils import get_config, get_logger

logger = get_logger(__name__)

# ─── App Setup ───

app = FastAPI(
    title="Demand Planner API",
    description="Supply chain demand forecasting and anomaly detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ───

preprocessor = DemandDataPreprocessor()
_data: Optional[pd.DataFrame] = None


def get_data() -> pd.DataFrame:
    global _data
    if _data is None:
        data_path = os.getenv("DATA_PATH", "sample_data/demand_data.csv")
        if not Path(data_path).exists():
            # Generate sample data
            from sample_data.generate import generate_sample_data
            generate_sample_data()
        raw = preprocessor.load(data_path)
        _data = preprocessor.preprocess(raw)
    return _data


# ─── Request/Response Models ───

class ForecastRequest(BaseModel):
    sku_id: str
    horizon_days: int = Field(default=30, ge=7, le=90)
    model: str = Field(default="ensemble", pattern="^(arima|exponential_smoothing|prophet|ensemble)$")
    warehouse: Optional[str] = None


class AnomalyRequest(BaseModel):
    sku_id: str
    threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    warehouse: Optional[str] = None


class DataEditRequest(BaseModel):
    sku_id: str
    date: str
    new_demand: float = Field(ge=0)


# ─── Endpoints ───

@app.get("/")
def root():
    return {"status": "ok", "service": "Demand Planner API", "version": "1.0.0"}


@app.get("/api/skus")
def list_skus():
    """List all available SKUs."""
    data = get_data()
    skus = preprocessor.list_skus(data)
    return {"skus": skus, "count": len(skus)}


@app.get("/api/summary")
def data_summary():
    """Get dataset summary statistics."""
    data = get_data()
    summary = preprocessor.summary(data)
    return summary


@app.get("/api/history/{sku_id}")
def get_history(sku_id: str, warehouse: Optional[str] = None, last_n_days: Optional[int] = None):
    """Get historical demand data for a SKU."""
    data = get_data()
    series = preprocessor.get_sku_series(data, sku_id, warehouse)

    if series.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {sku_id}")

    if last_n_days:
        series = series.tail(last_n_days)

    records = [
        {"date": str(idx.date()), "demand": round(float(row["demand"]), 2)}
        for idx, row in series.iterrows()
    ]

    demands = series["demand"]
    recent_7 = demands.tail(7).mean()
    prev_7 = demands.iloc[-14:-7].mean() if len(demands) > 14 else demands.mean()
    change_pct = round((recent_7 - prev_7) / prev_7 * 100, 1) if prev_7 > 0 else 0

    # Day of week averages
    dow_data = series.copy()
    dow_data["dow"] = dow_data.index.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = dow_data.groupby("dow")["demand"].mean()
    dow_stats = [
        {"day": day_names[i], "avg": round(float(dow_avg.get(i, 0)), 2)}
        for i in range(7)
    ]

    return {
        "sku_id": sku_id,
        "data": records,
        "stats": {
            "mean": round(float(demands.mean()), 2),
            "std": round(float(demands.std()), 2),
            "min": round(float(demands.min()), 2),
            "max": round(float(demands.max()), 2),
            "total": round(float(demands.sum()), 2),
            "points": len(demands),
            "recent_7_avg": round(float(recent_7), 2),
            "change_pct": change_pct,
        },
        "dow_pattern": dow_stats,
        "date_range": {
            "start": str(series.index[0].date()),
            "end": str(series.index[-1].date()),
        },
    }


@app.post("/api/forecast")
def run_forecast(request: ForecastRequest):
    """Run demand forecast for a SKU."""
    data = get_data()
    series = preprocessor.get_sku_series(data, request.sku_id, request.warehouse)

    if series.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.sku_id}")

    if len(series) < 14:
        raise HTTPException(status_code=400, detail=f"Need at least 14 data points, got {len(series)}")

    cfg = get_config()["forecasting"]
    results = {}

    models_to_run = (
        ["arima", "exponential_smoothing", "prophet"]
        if request.model == "ensemble"
        else [request.model]
    )

    errors = {}
    for name in models_to_run:
        model_cfg = cfg["models"].get(name, {})
        if not model_cfg.get("enabled", True):
            continue

        try:
            if name == "arima":
                m = ARIMAModel()
            elif name == "exponential_smoothing":
                m = ExponentialSmoothingModel()
            elif name == "prophet":
                m = ProphetModel()
            else:
                continue

            m.fit(series)
            pred = m.predict(request.horizon_days)
            results[name] = pred
        except Exception as e:
            errors[name] = str(e)
            logger.warning(f"Model {name} failed: {e}")

    if not results:
        raise HTTPException(
            status_code=500,
            detail=f"All models failed: {errors}"
        )

    # Build forecast dates
    last_date = series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=request.horizon_days)

    # Ensemble or single
    if request.model == "ensemble" and len(results) > 1:
        weights = cfg.get("ensemble", {}).get("weights", {})
        total_weight = sum(weights.get(n, 1.0) for n in results)
        forecast_vals = sum(
            results[n]["forecast"].values * (weights.get(n, 1.0) / total_weight)
            for n in results
        )
        lower_vals = np.minimum.reduce([results[n]["lower_bound"].values for n in results])
        upper_vals = np.maximum.reduce([results[n]["upper_bound"].values for n in results])
        model_used = f"ensemble ({', '.join(results.keys())})"
    else:
        name = list(results.keys())[0]
        forecast_vals = results[name]["forecast"].values
        lower_vals = results[name]["lower_bound"].values
        upper_vals = results[name]["upper_bound"].values
        model_used = name

    daily_forecast = [
        {
            "date": str(forecast_dates[i].date()),
            "forecast": round(float(forecast_vals[i]), 2),
            "lower": round(float(lower_vals[i]), 2),
            "upper": round(float(upper_vals[i]), 2),
        }
        for i in range(len(forecast_dates))
    ]

    total_forecast = float(np.sum(forecast_vals))
    avg_forecast = float(np.mean(forecast_vals))

    return {
        "sku_id": request.sku_id,
        "model": model_used,
        "horizon_days": request.horizon_days,
        "data": daily_forecast,
        "summary": {
            "avg_daily": round(avg_forecast, 2),
            "total": round(total_forecast, 2),
            "safety_stock_15": round(total_forecast * 0.15, 2),
            "reorder_point_7d": round(avg_forecast * 7, 2),
            "confidence_lower": round(float(np.min(lower_vals)), 2),
            "confidence_upper": round(float(np.max(upper_vals)), 2),
        },
        "models_used": list(results.keys()),
        "errors": errors if errors else None,
    }


@app.post("/api/anomalies")
def detect_anomalies(request: AnomalyRequest):
    """Detect demand anomalies for a SKU."""
    data = get_data()
    series = preprocessor.get_sku_series(data, request.sku_id, request.warehouse)

    if series.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.sku_id}")

    values = series["demand"].values.astype(float)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    if std_val == 0:
        return {"sku_id": request.sku_id, "anomalies": [], "count": 0, "stats": {"mean": mean_val, "std": 0}}

    anomalies = []
    for i, val in enumerate(values):
        z = abs((val - mean_val) / std_val)
        if z > request.threshold:
            deviation = round((val - mean_val) / mean_val * 100, 1)
            anomalies.append({
                "date": str(series.index[i].date()),
                "demand": round(float(val), 2),
                "z_score": round(float(z), 2),
                "deviation_pct": deviation,
                "type": "spike" if val > mean_val else "drop",
                "severity": "high" if abs(deviation) > 100 else "medium",
            })

    spikes = sum(1 for a in anomalies if a["type"] == "spike")
    drops = sum(1 for a in anomalies if a["type"] == "drop")

    return {
        "sku_id": request.sku_id,
        "threshold": request.threshold,
        "anomalies": anomalies,
        "count": len(anomalies),
        "spikes": spikes,
        "drops": drops,
        "stats": {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "upper_bound": round(mean_val + request.threshold * std_val, 2),
            "lower_bound": round(max(0, mean_val - request.threshold * std_val), 2),
        },
    }


@app.post("/api/edit")
def edit_data_point(request: DataEditRequest):
    """Edit a single demand data point. Updates the in-memory dataset."""
    global _data
    data = get_data()

    mask = (
        (data[preprocessor.sku_col] == request.sku_id)
        & (data[preprocessor.date_col].dt.strftime("%Y-%m-%d") == request.date)
    )

    if mask.sum() == 0:
        raise HTTPException(status_code=404, detail=f"No data found for {request.sku_id} on {request.date}")

    old_value = float(data.loc[mask, preprocessor.demand_col].iloc[0])
    data.loc[mask, preprocessor.demand_col] = request.new_demand
    _data = data

    return {
        "status": "updated",
        "sku_id": request.sku_id,
        "date": request.date,
        "old_demand": round(old_value, 2),
        "new_demand": request.new_demand,
    }


@app.post("/api/reset")
def reset_data():
    """Reload data from disk, discarding all edits."""
    global _data
    _data = None
    get_data()
    return {"status": "reset", "message": "Data reloaded from source"}


@app.get("/api/compare")
def compare_skus(skus: str, metric: str = "weekly_avg"):
    """
    Compare multiple SKUs.
    Pass SKUs as comma-separated string, e.g., ?skus=SKU-1001,SKU-1003
    """
    data = get_data()
    sku_list = [s.strip() for s in skus.split(",") if s.strip()]

    if not sku_list:
        raise HTTPException(status_code=400, detail="Provide at least one SKU")

    comparison = {}
    for sku_id in sku_list:
        series = preprocessor.get_sku_series(data, sku_id)
        if series.empty:
            continue

        demands = series["demand"]
        weekly = demands.resample("W").mean()

        comparison[sku_id] = {
            "weekly_data": [
                {"date": str(idx.date()), "avg": round(float(val), 2)}
                for idx, val in weekly.items()
            ],
            "stats": {
                "mean": round(float(demands.mean()), 2),
                "std": round(float(demands.std()), 2),
                "min": round(float(demands.min()), 2),
                "max": round(float(demands.max()), 2),
                "trend": "up" if demands.tail(7).mean() > demands.head(7).mean() else "down",
            },
        }

    return {"skus": comparison}


@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload a new CSV data file."""
    global _data
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")

    contents = await file.read()
    tmp_path = Path("/tmp/uploaded_demand.csv")
    tmp_path.write_bytes(contents)

    try:
        raw = preprocessor.load(str(tmp_path))
        _data = preprocessor.preprocess(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {e}")

    summary = preprocessor.summary(_data)
    return {"status": "uploaded", "filename": file.filename, "summary": summary}


# ─── Health Check ───

@app.get("/health")
def health():
    return {"status": "healthy"}


# ─── AI Chat (LangChain Agent with GPT-4o) ───

_agent = None
_chat_history = []


def get_agent():
    """Lazily initialize the LangChain demand agent."""
    global _agent
    if _agent is None:
        # Ensure data is loaded for tools
        get_data()
        from src.tools.forecasting import set_demand_data
        set_demand_data(get_data())

        from src.agents.demand_agent import create_demand_agent
        _agent = create_demand_agent(verbose=True)
        logger.info("LangChain agent initialized for chat")
    return _agent


class ChatRequest(BaseModel):
    message: str
    sku_id: Optional[str] = None
    reset_history: bool = False


class ChatResponse(BaseModel):
    response: str
    sku_id: Optional[str] = None


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a message to the AI demand planning assistant.
    Uses GPT-4o via LangChain with access to all forecasting tools.
    The agent can run forecasts, detect anomalies, query data,
    and generate reports based on your questions.
    """
    global _chat_history

    if request.reset_history:
        _chat_history = []
        return ChatResponse(response="Chat history cleared. How can I help?", sku_id=request.sku_id)

    # Make sure tools have current data
    from src.tools.forecasting import set_demand_data
    set_demand_data(get_data())

    agent = get_agent()

    # Add SKU context to the message if provided
    user_message = request.message
    if request.sku_id:
        user_message = f"[Current SKU context: {request.sku_id}] {request.message}"

    try:
        result = agent.invoke({
            "input": user_message,
            "chat_history": _chat_history,
        })

        response_text = result.get("output", "I wasn't able to generate a response. Please try rephrasing your question.")

        # Update chat history
        from langchain_core.messages import HumanMessage, AIMessage
        _chat_history.append(HumanMessage(content=request.message))
        _chat_history.append(AIMessage(content=response_text))

        # Keep history manageable (last 20 exchanges)
        if len(_chat_history) > 40:
            _chat_history = _chat_history[-40:]

        return ChatResponse(response=response_text, sku_id=request.sku_id)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        error_msg = str(e)

        # Provide helpful error messages
        if "API key" in error_msg or "401" in error_msg:
            return ChatResponse(
                response="API key error. Make sure your OPENAI_API_KEY is set correctly in .env or your shell environment.",
                sku_id=request.sku_id,
            )
        if "quota" in error_msg.lower() or "429" in error_msg:
            return ChatResponse(
                response="Rate limit or quota exceeded. Please wait a moment and try again, or check your OpenAI billing.",
                sku_id=request.sku_id,
            )

        return ChatResponse(
            response=f"Something went wrong: {error_msg}. Try a simpler question or check the server logs.",
            sku_id=request.sku_id,
        )


@app.post("/api/chat/reset")
def reset_chat():
    """Clear chat history."""
    global _chat_history
    _chat_history = []
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
