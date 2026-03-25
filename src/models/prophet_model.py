"""Prophet-based demand forecasting model."""

import numpy as np
import pandas as pd

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class ProphetModel:
    """Meta Prophet model for time series forecasting with seasonality."""

    def __init__(self):
        cfg = get_config()["forecasting"]["models"]["prophet"]
        self.yearly_seasonality = cfg["yearly_seasonality"]
        self.weekly_seasonality = cfg["weekly_seasonality"]
        self.daily_seasonality = cfg["daily_seasonality"]
        self.changepoint_prior_scale = cfg["changepoint_prior_scale"]
        self._model = None
        self._fitted = False

    def fit(self, series: pd.DataFrame, demand_col: str = "demand") -> "ProphetModel":
        """
        Fit Prophet on historical demand.

        Args:
            series: DataFrame with DatetimeIndex and a demand column.
            demand_col: Name of the demand column.

        Returns:
            self
        """
        from prophet import Prophet

        # Prophet expects columns named 'ds' and 'y'
        prophet_df = pd.DataFrame(
            {"ds": series.index, "y": series[demand_col].values.astype(float)}
        )

        logger.info(f"Fitting Prophet on {len(prophet_df)} data points")

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        # Suppress Prophet's verbose output
        self._model.fit(prophet_df)
        self._fitted = True

        logger.info("Prophet model fitted successfully")
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Generate demand forecast.

        Args:
            horizon: Number of days to forecast.

        Returns:
            DataFrame with forecast, lower_bound, upper_bound columns.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        future = self._model.make_future_dataframe(periods=horizon)
        prediction = self._model.predict(future)

        # Take only the forecast horizon rows
        forecast_rows = prediction.tail(horizon)

        result = pd.DataFrame(
            {
                "forecast": np.maximum(forecast_rows["yhat"].values, 0),
                "lower_bound": np.maximum(forecast_rows["yhat_lower"].values, 0),
                "upper_bound": np.maximum(forecast_rows["yhat_upper"].values, 0),
            }
        )

        return result

    @property
    def is_fitted(self) -> bool:
        return self._fitted
