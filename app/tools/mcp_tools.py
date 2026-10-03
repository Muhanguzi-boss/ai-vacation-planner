"""MCP adapter for the LangChain tool layer.

LangGraph and the model keep seeing the ordinary ``travel_weather`` StructuredTool built by
``create_weather_tool``; only the weather service behind it changes. ``MCPWeatherService``
fulfils that service contract by calling the MCP weather server's ``get_forecast`` tool
through ``MCPClient``, so no MCP objects reach LangGraph or the model, and failures surface
through the existing ``Weather lookup failed`` tool error.
"""

from typing import Any

from langchain_core.tools import StructuredTool

from app.services.mcp_client import MCPClient
from app.tools.travel_tools import create_weather_tool

WEATHER_MCP_TOOL = "get_forecast"


class MCPWeatherService:
    def __init__(self, client: Any | None = None) -> None:
        self.client = client or MCPClient()

    def get_weather(self, location: str) -> dict[str, Any]:
        return self.client.call_tool(WEATHER_MCP_TOOL, {"location": location})


def create_mcp_weather_tool(client: Any | None = None) -> StructuredTool:
    return create_weather_tool(MCPWeatherService(client))
