"""Validation logic for demand data inputs."""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator


class ForecastRequest(BaseModel):
    """Validated forecast request."""

    sku_id: str
    horizon_days: int = 30
    warehouse: Optional[str] = None
    model: str = "ensemble"

    @field_validator("horizon_days")
    @classmethod
    def horizon_positive(cls, v: int) -> int:
        if v < 1 or v > 365:
            raise ValueError("horizon_days must be between 1 and 365")
        return v

    @field_validator("model")
    @classmethod
    def valid_model(cls, v: str) -> str:
        allowed = {"arima", "exponential_smoothing", "prophet", "ensemble"}
        if v not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return v


class DataValidator:
    """Validate demand DataFrames before processing."""

    @staticmethod
    def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
        """Check for required columns, return list of missing ones."""
        return [col for col in required if col not in df.columns]

    @staticmethod
    def validate_min_rows(df: pd.DataFrame, min_rows: int = 30) -> bool:
        """Ensure sufficient data points for forecasting."""
        return len(df) >= min_rows

    @staticmethod
    def validate_date_continuity(
        df: pd.DataFrame, date_col: str, max_gap_days: int = 7
    ) -> list[dict]:
        """Find gaps in the time series larger than max_gap_days."""
        if date_col not in df.columns:
            return []

        dates = pd.to_datetime(df[date_col]).sort_values()
        gaps = []
        for i in range(1, len(dates)):
            diff = (dates.iloc[i] - dates.iloc[i - 1]).days
            if diff > max_gap_days:
                gaps.append(
                    {
                        "from": str(dates.iloc[i - 1].date()),
                        "to": str(dates.iloc[i].date()),
                        "gap_days": diff,
                    }
                )
        return gaps
