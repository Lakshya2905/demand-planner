"""Exponential Smoothing (Holt-Winters) demand forecasting model."""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class ExponentialSmoothingModel:
    def __init__(self):
        cfg = get_config()["forecasting"]["models"]["exponential_smoothing"]
        self.trend = cfg["trend"]
        self.seasonal = cfg["seasonal"]
        self.seasonal_periods = cfg["seasonal_periods"]
        self._fitted = None

    def fit(self, series, demand_col="demand"):
        values = series[demand_col].values.astype(float)
        min_required = self.seasonal_periods * 2
        if len(values) < min_required:
            logger.warning(f"Only {len(values)} points, need {min_required}. Falling back to simple.")
            self.seasonal = None
        logger.info(f"Fitting ExponentialSmoothing (trend={self.trend}, seasonal={self.seasonal})")
        try:
            model = ExponentialSmoothing(values, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods if self.seasonal else None)
            self._fitted = model.fit(optimized=True)
            logger.info(f"ETS AIC: {self._fitted.aic:.2f}")
        except Exception as e:
            logger.error(f"ExponentialSmoothing fitting failed: {e}")
            raise
        return self

    def predict(self, horizon):
        if self._fitted is None:
            raise RuntimeError("Model not fitted.")
        forecast = np.array(self._fitted.forecast(horizon))
        residuals = np.array(self._fitted.resid)
        std = np.std(residuals)
        return pd.DataFrame({
            "forecast": np.maximum(forecast, 0),
            "lower_bound": np.maximum(forecast - 1.96 * std, 0),
            "upper_bound": np.maximum(forecast + 1.96 * std, 0),
        })

    @property
    def is_fitted(self):
        return self._fitted is not None
