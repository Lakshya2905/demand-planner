"""Main demand planning agent using LangChain and OpenAI."""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.tools import (
    forecast_demand,
    detect_anomalies,
    get_data_summary,
    list_available_skus,
    get_sku_history,
    generate_demand_report,
)
from src.utils import get_api_key, get_model_name, get_config, get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert supply chain demand planning assistant. Your role is to help
users understand, forecast, and optimize their inventory demand.

You have access to tools for:
- Getting data summaries and listing SKUs
- Viewing historical demand for specific SKUs
- Forecasting demand using statistical models (ARIMA, Exponential Smoothing, Prophet, or ensemble)
- Detecting demand anomalies (spikes and drops)
- Generating comprehensive demand planning reports

When answering questions:
1. Always check what data is available before making assumptions
2. Use the ensemble model by default for forecasts unless the user requests a specific model
3. When generating reports, first run forecast and anomaly detection, then combine results
4. Provide clear, actionable insights with your analysis
5. If data is insufficient, explain what is needed
6. Express numbers with appropriate precision (round to 2 decimal places)

Be concise but thorough. Always ground your answers in the actual data."""

TOOL_MAP = {
    "get_data_summary": get_data_summary,
    "list_available_skus": list_available_skus,
    "get_sku_history": get_sku_history,
    "forecast_demand": forecast_demand,
    "detect_anomalies": detect_anomalies,
    "generate_demand_report": generate_demand_report,
}


class DemandAgent:
    def __init__(self, verbose=True):
        cfg = get_config()["agent"]
        self.llm = ChatOpenAI(
            model=get_model_name(),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 4096),
            api_key=get_api_key(),
        )
        self.tools = [
            get_data_summary, list_available_skus, get_sku_history,
            forecast_demand, detect_anomalies, generate_demand_report,
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.verbose = verbose
        self.max_iterations = 10

    def invoke(self, inputs):
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        chat_history = inputs.get("chat_history", [])
        messages.extend(chat_history)
        messages.append(HumanMessage(content=inputs["input"]))

        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"Agent iteration {iteration + 1}")
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return {"output": response.content}

            from langchain_core.messages import ToolMessage
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if self.verbose:
                    logger.info(f"Calling tool: {tool_name}({tool_args})")
                tool_fn = TOOL_MAP.get(tool_name)
                if tool_fn is None:
                    result = f"Error: Unknown tool '{tool_name}'"
                else:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        result = f"Error running {tool_name}: {e}"
                        logger.error(result)
                if self.verbose:
                    display = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    logger.info(f"Tool result: {display}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        return {"output": "Reached maximum iterations. Please try a simpler query."}


def create_demand_agent(verbose=True):
    agent = DemandAgent(verbose=verbose)
    logger.info("Demand planning agent created successfully")
    return agent
