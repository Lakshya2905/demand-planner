"""LangChain tool for demand anomaly detection."""

import json
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.data import DemandDataPreprocessor
from src.utils import get_config, get_logger

logger = get_logger(__name__)


def _detect_zscore(values: np.ndarray, threshold: float) -> list[int]:
    """Find indices where z-score exceeds threshold."""
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []
    z_scores = np.abs((values - mean) / std)
    return list(np.where(z_scores > threshold)[0])


def _detect_iqr(values: np.ndarray, multiplier: float) -> list[int]:
    """Find indices outside the IQR-based bounds."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return list(np.where((values < lower) | (values > upper))[0])


@tool
def detect_anomalies(
    sku_id: str,
    warehouse: Optional[str] = None,
    lookback_days: Optional[int] = None,
) -> str:
    """
    Detect anomalies in demand data for a specific SKU.

    Identifies unusual demand spikes, drops, and deviations from normal patterns
    using statistical methods (z-score or IQR).

    Args:
        sku_id: The product/SKU identifier to analyze.
        warehouse: Optional warehouse filter.
        lookback_days: Number of historical days to analyze. Uses config default if not specified.

    Returns:
        JSON string with detected anomalies including dates, values,
        severity, and type (spike or drop).
    """
    from src.tools.forecasting import _get_data

    cfg = get_config()["anomaly_detection"]
    method = cfg["method"]
    lookback = lookback_days or cfg["lookback_days"]

    preprocessor = DemandDataPreprocessor()
    series = preprocessor.get_sku_series(_get_data(), sku_id, warehouse)

    if len(series) < cfg["min_data_points"]:
        return json.dumps(
            {
                "error": f"Need at least {cfg['min_data_points']} data points for anomaly detection, "
                         f"got {len(series)}."
            }
        )

    # Apply lookback window
    if lookback and len(series) > lookback:
        series = series.tail(lookback)

    values = series["demand"].values.astype(float)
    mean_val = float(np.mean(values))

    # Detect anomalies
    if method == "zscore":
        anomaly_indices = _detect_zscore(values, cfg["zscore_threshold"])
    elif method == "iqr":
        anomaly_indices = _detect_iqr(values, cfg["iqr_multiplier"])
    else:
        return json.dumps({"error": f"Unknown anomaly method: {method}"})

    # Build anomaly details
    anomalies = []
    for idx in anomaly_indices:
        val = float(values[idx])
        date = str(series.index[idx].date()) if hasattr(series.index[idx], "date") else str(series.index[idx])

        deviation = (val - mean_val) / mean_val * 100 if mean_val > 0 else 0

        anomalies.append(
            {
                "date": date,
                "demand": round(val, 2),
                "expected_mean": round(mean_val, 2),
                "deviation_pct": round(deviation, 1),
                "type": "spike" if val > mean_val else "drop",
                "severity": "high" if abs(deviation) > 100 else "medium",
            }
        )

    output = {
        "sku_id": sku_id,
        "warehouse": warehouse,
        "method": method,
        "lookback_days": lookback,
        "data_points_analyzed": len(values),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "summary": {
            "mean_demand": round(mean_val, 2),
            "std_demand": round(float(np.std(values)), 2),
            "spike_count": sum(1 for a in anomalies if a["type"] == "spike"),
            "drop_count": sum(1 for a in anomalies if a["type"] == "drop"),
        },
    }

    return json.dumps(output, indent=2)
