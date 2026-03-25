"""LangChain tool for querying and summarizing demand data."""

import json
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from src.data import DemandDataPreprocessor
from src.utils import get_logger

logger = get_logger(__name__)


@tool
def get_data_summary() -> str:
    """
    Get a summary of the currently loaded demand dataset.

    Returns a JSON overview including row count, date range,
    number of SKUs, and demand statistics.
    """
    from src.tools.forecasting import _get_data

    preprocessor = DemandDataPreprocessor()
    summary = preprocessor.summary(_get_data())
    return json.dumps(summary, indent=2)


@tool
def list_available_skus() -> str:
    """
    List all available SKU identifiers in the dataset.

    Returns a JSON array of SKU IDs that can be used for
    forecasting and anomaly detection.
    """
    from src.tools.forecasting import _get_data

    preprocessor = DemandDataPreprocessor()
    skus = preprocessor.list_skus(_get_data())
    return json.dumps({"sku_count": len(skus), "skus": skus})


@tool
def get_sku_history(
    sku_id: str,
    last_n_days: Optional[int] = None,
    warehouse: Optional[str] = None,
) -> str:
    """
    Get historical demand data for a specific SKU.

    Args:
        sku_id: The product/SKU identifier.
        last_n_days: Only return the last N days of data. Returns all if not specified.
        warehouse: Optional warehouse filter.

    Returns:
        JSON with date/demand pairs and basic statistics for the SKU.
    """
    from src.tools.forecasting import _get_data

    preprocessor = DemandDataPreprocessor()
    series = preprocessor.get_sku_series(_get_data(), sku_id, warehouse)

    if series.empty:
        return json.dumps({"error": f"No data found for SKU {sku_id}"})

    if last_n_days:
        series = series.tail(last_n_days)

    records = [
        {"date": str(idx.date()), "demand": round(float(row["demand"]), 2)}
        for idx, row in series.iterrows()
    ]

    output = {
        "sku_id": sku_id,
        "data_points": len(records),
        "date_range": {
            "start": records[0]["date"] if records else None,
            "end": records[-1]["date"] if records else None,
        },
        "stats": {
            "mean": round(float(series["demand"].mean()), 2),
            "min": round(float(series["demand"].min()), 2),
            "max": round(float(series["demand"].max()), 2),
            "trend": "increasing"
            if len(series) > 7
            and series["demand"].tail(7).mean() > series["demand"].head(7).mean()
            else "decreasing",
        },
        "recent_data": records[-14:],  # Last 14 days for display
    }

    return json.dumps(output, indent=2)
