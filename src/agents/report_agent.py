"""Specialized agent for batch report generation."""

import json
from typing import Optional

import pandas as pd

from src.agents.demand_agent import create_demand_agent
from src.data import DemandDataPreprocessor
from src.tools.forecasting import set_demand_data
from src.utils import get_logger

logger = get_logger(__name__)


class ReportAgent:
    """Agent that generates demand reports for one or more SKUs."""

    def __init__(self, data_path: str):
        preprocessor = DemandDataPreprocessor()
        raw = preprocessor.load(data_path)
        self.data = preprocessor.preprocess(raw)
        set_demand_data(self.data)
        self.agent = create_demand_agent(verbose=False)

    def generate_single_report(
        self, sku_id: str, horizon_days: int = 30
    ) -> str:
        """Generate a full demand report for a single SKU."""
        query = (
            f"Generate a complete demand planning report for SKU '{sku_id}' "
            f"with a {horizon_days}-day forecast horizon. "
            f"First run the forecast, then detect anomalies, then generate the report."
        )

        result = self.agent.invoke({"input": query})
        return result.get("output", "Report generation failed.")

    def generate_batch_reports(
        self, sku_ids: Optional[list[str]] = None, horizon_days: int = 30
    ) -> dict[str, str]:
        """Generate reports for multiple SKUs."""
        preprocessor = DemandDataPreprocessor()

        if sku_ids is None:
            sku_ids = preprocessor.list_skus(self.data)

        reports = {}
        for sku_id in sku_ids:
            logger.info(f"Generating report for {sku_id}...")
            try:
                reports[sku_id] = self.generate_single_report(sku_id, horizon_days)
            except Exception as e:
                logger.error(f"Failed to generate report for {sku_id}: {e}")
                reports[sku_id] = f"Error: {e}"

        return reports
