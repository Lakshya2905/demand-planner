"""Generate realistic sample supply chain demand data for testing."""

import numpy as np
import pandas as pd


def generate_sample_data(
    num_skus: int = 5,
    num_days: int = 365,
    start_date: str = "2025-01-01",
    output_path: str = "sample_data/demand_data.csv",
) -> pd.DataFrame:
    """
    Generate synthetic demand data with realistic patterns.

    Includes: weekly seasonality, trend, noise, and occasional anomalies.
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")

    sku_configs = {
        "SKU-1001": {"base": 150, "trend": 0.3, "noise": 20, "warehouse": "warehouse-east"},
        "SKU-1002": {"base": 80, "trend": -0.1, "noise": 15, "warehouse": "warehouse-east"},
        "SKU-1003": {"base": 300, "trend": 0.5, "noise": 40, "warehouse": "warehouse-west"},
        "SKU-1004": {"base": 50, "trend": 0.0, "noise": 10, "warehouse": "warehouse-west"},
        "SKU-1005": {"base": 200, "trend": 0.2, "noise": 30, "warehouse": "warehouse-central"},
    }

    rows = []
    for sku_id, cfg in list(sku_configs.items())[:num_skus]:
        for i, date in enumerate(dates):
            # Base demand + trend
            demand = cfg["base"] + cfg["trend"] * i

            # Weekly seasonality (higher on Mon-Fri, lower on weekends)
            day_of_week = date.dayofweek
            if day_of_week < 5:
                demand *= 1.1
            else:
                demand *= 0.75

            # Monthly seasonality
            demand *= 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)

            # Random noise
            demand += np.random.normal(0, cfg["noise"])

            # Inject occasional anomalies (about 2% of days)
            if np.random.random() < 0.02:
                if np.random.random() < 0.5:
                    demand *= np.random.uniform(2.0, 3.0)  # Spike
                else:
                    demand *= np.random.uniform(0.1, 0.3)  # Drop

            demand = max(0, round(demand, 2))

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "sku_id": sku_id,
                    "demand": demand,
                    "warehouse": cfg["warehouse"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows of sample data -> {output_path}")
    return df


if __name__ == "__main__":
    generate_sample_data()
