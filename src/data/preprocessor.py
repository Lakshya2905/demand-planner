"""Data loading and preprocessing for demand data."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class DemandDataPreprocessor:
    """Load, validate, and preprocess supply chain demand data."""

    def __init__(self):
        cfg = get_config()["data"]
        self.date_col = cfg["date_column"]
        self.demand_col = cfg["demand_column"]
        self.sku_col = cfg["sku_column"]
        self.warehouse_col = cfg["warehouse_column"]

    def load(self, filepath: str | Path) -> pd.DataFrame:
        """Load demand data from CSV, Excel, or Parquet."""
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        logger.info(f"Loading data from {filepath}")

        if suffix == ".csv":
            df = pd.read_csv(filepath)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        elif suffix == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare demand data for forecasting."""
        df = df.copy()

        # Parse dates
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.sort_values(self.date_col)

        # Handle missing demand values
        if self.demand_col in df.columns:
            missing_count = df[self.demand_col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing demand values, forward-filling")
                df[self.demand_col] = df[self.demand_col].ffill().bfill()

            # Ensure non-negative demand
            neg_count = (df[self.demand_col] < 0).sum()
            if neg_count > 0:
                logger.warning(f"Found {neg_count} negative demand values, clipping to 0")
                df[self.demand_col] = df[self.demand_col].clip(lower=0)

        # Remove duplicate date entries per SKU
        group_cols = [c for c in [self.date_col, self.sku_col] if c in df.columns]
        if group_cols:
            before = len(df)
            df = df.drop_duplicates(subset=group_cols, keep="last")
            if len(df) < before:
                logger.info(f"Removed {before - len(df)} duplicate rows")

        return df

    def get_sku_series(
        self, df: pd.DataFrame, sku_id: str, warehouse: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract time series for a specific SKU (and optionally warehouse)."""
        mask = df[self.sku_col] == sku_id
        if warehouse and self.warehouse_col in df.columns:
            mask &= df[self.warehouse_col] == warehouse

        series = df.loc[mask, [self.date_col, self.demand_col]].copy()
        series = series.set_index(self.date_col).sort_index()

        logger.info(f"Extracted series for {sku_id}: {len(series)} data points")
        return series

    def list_skus(self, df: pd.DataFrame) -> list[str]:
        """Return all unique SKU identifiers."""
        if self.sku_col in df.columns:
            return sorted(df[self.sku_col].unique().tolist())
        return []

    def summary(self, df: pd.DataFrame) -> dict:
        """Generate a summary of the demand dataset."""
        info = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "date_range": None,
            "sku_count": 0,
            "demand_stats": {},
        }

        if self.date_col in df.columns:
            info["date_range"] = {
                "start": str(df[self.date_col].min()),
                "end": str(df[self.date_col].max()),
            }

        if self.sku_col in df.columns:
            info["sku_count"] = df[self.sku_col].nunique()

        if self.demand_col in df.columns:
            info["demand_stats"] = {
                "mean": float(df[self.demand_col].mean()),
                "median": float(df[self.demand_col].median()),
                "std": float(df[self.demand_col].std()),
                "min": float(df[self.demand_col].min()),
                "max": float(df[self.demand_col].max()),
            }

        return info
