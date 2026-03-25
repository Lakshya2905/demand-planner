"""LangChain chain for structured forecasting workflow."""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.utils import get_api_key, get_model_name


def create_forecast_analysis_chain():
    """
    Create a chain that takes raw forecast data and produces
    a human-readable analysis with actionable recommendations.
    """
    llm = ChatOpenAI(
        model=get_model_name(),
        temperature=0.2,
        max_tokens=2048,
        api_key=get_api_key(),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a supply chain analyst. Analyze the forecast data and provide "
                "concise, actionable insights. Focus on trends, risks, and recommendations "
                "for inventory management. Be specific with numbers.",
            ),
            (
                "human",
                "Analyze this demand forecast data and provide actionable insights:\n\n"
                "SKU: {sku_id}\n"
                "Forecast Data: {forecast_json}\n"
                "Historical Context: {history_json}\n\n"
                "Provide: 1) Trend analysis, 2) Risk assessment, 3) Inventory recommendations",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def create_anomaly_analysis_chain():
    """
    Create a chain that interprets anomaly detection results
    and suggests root causes and corrective actions.
    """
    llm = ChatOpenAI(
        model=get_model_name(),
        temperature=0.2,
        max_tokens=2048,
        api_key=get_api_key(),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a supply chain analyst specializing in demand pattern analysis. "
                "Interpret anomaly detection results and suggest likely root causes "
                "and corrective actions for supply chain managers.",
            ),
            (
                "human",
                "Analyze these demand anomalies and suggest root causes:\n\n"
                "SKU: {sku_id}\n"
                "Anomaly Data: {anomaly_json}\n\n"
                "For each anomaly, suggest: 1) Likely cause, 2) Recommended action, "
                "3) Impact on inventory planning",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain
