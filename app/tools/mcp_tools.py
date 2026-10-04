"""MCP adapter for the LangChain tool layer.

LangGraph and the model keep seeing the ordinary ``travel_weather`` StructuredTool built by
``create_weather_tool``; only the weather service behind it changes. ``MCPWeatherService``
fulfils that service contract by calling the MCP weather server's ``get_forecast`` tool
through ``MCPClient``, so no MCP objects reach LangGraph or the model, and failures surface
through the existing ``Weather lookup failed`` tool error.

Weather is supplementary context, so when a fallback is supplied an MCP failure is logged
and answered with the fallback's result (labelled as such) instead of failing the itinerary.
"""

import logging
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from app.services.mcp_client import MCPClient, MCPClientError
from app.tools.travel_tools import create_weather_tool

logger = logging.getLogger(__name__)

WEATHER_MCP_TOOL = "get_forecast"
FALLBACK_NOTE = "Live forecast unavailable; this is approximate local weather information."


class MCPWeatherService:
    def __init__(self, client: Any | None = None, fallback: Callable[[str], dict] | None = None) -> None:
        self.client = client or MCPClient()
        self.fallback = fallback

    def get_weather(self, location: str) -> dict[str, Any]:
        try:
            return self.client.call_tool(WEATHER_MCP_TOOL, {"location": location})
        except MCPClientError as exc:
            if self.fallback is None:
                raise
            logger.warning("MCP weather lookup for %r failed (%s); using local weather fallback", location, exc)
            return {**self.fallback(location), "source": "local fallback", "note": FALLBACK_NOTE}


def create_mcp_weather_tool(
    client: Any | None = None,
    fallback: Callable[[str], dict] | None = None,
) -> StructuredTool:
    return create_weather_tool(MCPWeatherService(client, fallback))
