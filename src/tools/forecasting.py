"""LangChain tool for demand forecasting."""

import json
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from src.data import DemandDataPreprocessor, ForecastRequest
from src.models import ARIMAModel, ExponentialSmoothingModel, ProphetModel
from src.utils import get_config, get_logger

logger = get_logger(__name__)

# Module-level data store (set by the agent at startup)
_demand_data: Optional[pd.DataFrame] = None


def set_demand_data(df: pd.DataFrame) -> None:
    """Set the demand DataFrame that tools operate on."""
    global _demand_data
    _demand_data = df


def _get_data() -> pd.DataFrame:
    if _demand_data is None:
        raise RuntimeError("No demand data loaded. Call set_demand_data() first.")
    return _demand_data


@tool
def forecast_demand(
    sku_id: str,
    horizon_days: int = 30,
    model_name: str = "ensemble",
    warehouse: Optional[str] = None,
) -> str:
    """
    Generate a demand forecast for a specific SKU.

    Args:
        sku_id: The product/SKU identifier to forecast.
        horizon_days: Number of days ahead to forecast (1 to 365).
        model_name: Model to use. Options: arima, exponential_smoothing, prophet, ensemble.
        warehouse: Optional warehouse filter.

    Returns:
        JSON string with the forecast results including predicted demand,
        confidence intervals, and model information.
    """
    request = ForecastRequest(
        sku_id=sku_id,
        horizon_days=horizon_days,
        warehouse=warehouse,
        model=model_name,
    )

    preprocessor = DemandDataPreprocessor()
    series = preprocessor.get_sku_series(_get_data(), request.sku_id, request.warehouse)

    if len(series) < 14:
        return json.dumps(
            {"error": f"Insufficient data for {sku_id}. Need at least 14 data points, got {len(series)}."}
        )

    cfg = get_config()["forecasting"]
    results = {}

    # Run requested models
    models_to_run = (
        ["arima", "exponential_smoothing", "prophet"]
        if request.model == "ensemble"
        else [request.model]
    )

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
            logger.warning(f"Model {name} failed: {e}")

    if not results:
        return json.dumps({"error": "All forecasting models failed."})

    # Ensemble or single model output
    if request.model == "ensemble" and len(results) > 1:
        weights = cfg.get("ensemble", {}).get("weights", {})
        total_weight = sum(weights.get(n, 1.0) for n in results)
        ensemble_forecast = sum(
            results[n]["forecast"] * (weights.get(n, 1.0) / total_weight)
            for n in results
        )
        ensemble_lower = min(results[n]["lower_bound"].min() for n in results)
        ensemble_upper = max(results[n]["upper_bound"].max() for n in results)

        output = {
            "sku_id": request.sku_id,
            "horizon_days": request.horizon_days,
            "model": "ensemble",
            "models_used": list(results.keys()),
            "forecast_mean": round(float(ensemble_forecast.mean()), 2),
            "forecast_total": round(float(ensemble_forecast.sum()), 2),
            "daily_forecast": [round(float(v), 2) for v in ensemble_forecast],
            "confidence_lower": round(float(ensemble_lower), 2),
            "confidence_upper": round(float(ensemble_upper), 2),
        }
    else:
        name = list(results.keys())[0]
        pred = results[name]
        output = {
            "sku_id": request.sku_id,
            "horizon_days": request.horizon_days,
            "model": name,
            "forecast_mean": round(float(pred["forecast"].mean()), 2),
            "forecast_total": round(float(pred["forecast"].sum()), 2),
            "daily_forecast": [round(float(v), 2) for v in pred["forecast"]],
            "confidence_lower": round(float(pred["lower_bound"].min()), 2),
            "confidence_upper": round(float(pred["upper_bound"].max()), 2),
        }

    return json.dumps(output, indent=2)
