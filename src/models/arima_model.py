"""ARIMA-based demand forecasting model."""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class ARIMAModel:
    def __init__(self):
        cfg = get_config()["forecasting"]["models"]["arima"]
        self.order = tuple(cfg["order"])
        self.seasonal_order = tuple(cfg["seasonal_order"])
        self._fitted = None

    def fit(self, series, demand_col="demand"):
        values = series[demand_col].values.astype(float)
        logger.info(f"Fitting SARIMA{self.order}x{self.seasonal_order} on {len(values)} points")
        try:
            model = SARIMAX(values, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            self._fitted = model.fit(disp=False, maxiter=200)
            logger.info(f"ARIMA AIC: {self._fitted.aic:.2f}")
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            raise
        return self

    def predict(self, horizon):
        if self._fitted is None:
            raise RuntimeError("Model not fitted.")
        forecast_result = self._fitted.get_forecast(steps=horizon)
        mean = np.array(forecast_result.predicted_mean)
        ci = np.array(forecast_result.conf_int(alpha=0.05))
        return pd.DataFrame({
            "forecast": np.maximum(mean, 0),
            "lower_bound": np.maximum(ci[:, 0], 0),
            "upper_bound": np.maximum(ci[:, 1], 0),
        })

    @property
    def is_fitted(self):
        return self._fitted is not None
