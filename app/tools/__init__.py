from langchain_core.tools import BaseTool

from app.tools.travel_tools import (
    travel_knowledge_search_tool,
    weather_tool,
    create_travel_knowledge_search_tool,
    create_weather_tool,
)


def default_travel_tools() -> list[BaseTool]:
    from app.core.config import settings

    if settings.MCP_WEATHER_ENABLED:
        from app.tools.mcp_tools import create_mcp_weather_tool

        return [travel_knowledge_search_tool, create_mcp_weather_tool()]
    return [travel_knowledge_search_tool, weather_tool]


__all__ = [
    "travel_knowledge_search_tool",
    "weather_tool",
    "create_travel_knowledge_search_tool",
    "create_weather_tool",
    "default_travel_tools",
]
