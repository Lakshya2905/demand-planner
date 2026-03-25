"""Tests for the demand planning tools and models."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DemandDataPreprocessor, DataValidator, ForecastRequest


# ---- Fixtures ----

@pytest.fixture
def sample_df():
    """Create a minimal demand DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    rows = []
    for date in dates:
        demand = 100 + 10 * np.sin(2 * np.pi * date.dayofweek / 7) + np.random.normal(0, 5)
        rows.append({"date": date, "sku_id": "TEST-001", "demand": max(0, demand), "warehouse": "wh-1"})
    return pd.DataFrame(rows)


@pytest.fixture
def preprocessor():
    return DemandDataPreprocessor()


# ---- Preprocessor Tests ----

class TestPreprocessor:
    def test_preprocess_sorts_by_date(self, preprocessor, sample_df):
        shuffled = sample_df.sample(frac=1, random_state=0)
        result = preprocessor.preprocess(shuffled)
        dates = result["date"].tolist()
        assert dates == sorted(dates)

    def test_preprocess_fills_missing(self, preprocessor, sample_df):
        sample_df.loc[5, "demand"] = None
        result = preprocessor.preprocess(sample_df)
        assert result["demand"].isna().sum() == 0

    def test_preprocess_clips_negative(self, preprocessor, sample_df):
        sample_df.loc[3, "demand"] = -50
        result = preprocessor.preprocess(sample_df)
        assert (result["demand"] >= 0).all()

    def test_get_sku_series(self, preprocessor, sample_df):
        processed = preprocessor.preprocess(sample_df)
        series = preprocessor.get_sku_series(processed, "TEST-001")
        assert len(series) == 90
        assert "demand" in series.columns

    def test_list_skus(self, preprocessor, sample_df):
        skus = preprocessor.list_skus(sample_df)
        assert skus == ["TEST-001"]

    def test_summary(self, preprocessor, sample_df):
        summary = preprocessor.summary(sample_df)
        assert summary["total_rows"] == 90
        assert summary["sku_count"] == 1
        assert "mean" in summary["demand_stats"]


# ---- Validator Tests ----

class TestValidator:
    def test_validate_columns_pass(self, sample_df):
        missing = DataValidator.validate_columns(sample_df, ["date", "demand", "sku_id"])
        assert missing == []

    def test_validate_columns_fail(self, sample_df):
        missing = DataValidator.validate_columns(sample_df, ["date", "nonexistent"])
        assert missing == ["nonexistent"]

    def test_validate_min_rows(self, sample_df):
        assert DataValidator.validate_min_rows(sample_df, 30) is True
        assert DataValidator.validate_min_rows(sample_df, 100) is False

    def test_validate_date_continuity(self, sample_df):
        gaps = DataValidator.validate_date_continuity(sample_df, "date", max_gap_days=2)
        assert gaps == []  # Daily data should have no gaps


# ---- ForecastRequest Tests ----

class TestForecastRequest:
    def test_valid_request(self):
        req = ForecastRequest(sku_id="SKU-001", horizon_days=30)
        assert req.sku_id == "SKU-001"
        assert req.model == "ensemble"

    def test_invalid_horizon(self):
        with pytest.raises(ValueError):
            ForecastRequest(sku_id="SKU-001", horizon_days=0)

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            ForecastRequest(sku_id="SKU-001", model="invalid_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
