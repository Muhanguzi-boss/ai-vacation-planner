from __future__ import annotations

from typing import Any, Mapping

from langchain_core.tools import StructuredTool


def _default_knowledge_service() -> Any:
    from app.services.knowledge_service import knowledge_service

    return knowledge_service


def _default_weather_service() -> Any:
    from app.services.weather_service import get_weather

    return get_weather


def _validate_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _search_knowledge(
    query: str,
    top_k: int = 5,
    knowledge_service: Any | None = None,
) -> Mapping[str, Any]:
    query = _validate_text(query, "query")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    service = knowledge_service or _default_knowledge_service()
    try:
        results = service.search(query, top_k=top_k)
    except Exception as exc:
        raise RuntimeError(f"Travel knowledge search failed: {exc}") from exc

    return {"query": query, "results": list(results)}


def _get_weather(
    location: str,
    weather_service: Any | None = None,
) -> Mapping[str, Any]:
    location = _validate_text(location, "location")
    service = weather_service or _default_weather_service()

    try:
        get_weather = getattr(service, "get_weather", None)
        forecast = get_weather(location) if get_weather else service(location)
    except Exception as exc:
        raise RuntimeError(f"Weather lookup failed: {exc}") from exc

    return {
        "location": location,
        "weather": forecast,
    }


def create_travel_knowledge_search_tool(
    knowledge_service: Any | None = None,
) -> StructuredTool:
    """Create a tool backed by the application's existing RAG service."""

    def search(query: str, top_k: int = 5) -> Mapping[str, Any]:
        return _search_knowledge(query, top_k, knowledge_service)

    return StructuredTool.from_function(
        func=search,
        name="travel_knowledge_search",
        description=(
            "Search the travel knowledge base for destination guidance, local "
            "customs, transportation, attractions, and planning constraints. "
            "Use this for factual travel context before proposing an itinerary."
        ),
    )


def create_weather_tool(weather_service: Any | None = None) -> StructuredTool:
    """Create a tool backed by the application's existing weather service."""

    def get_weather(
        location: str,
    ) -> Mapping[str, Any]:
        return _get_weather(location, weather_service)

    return StructuredTool.from_function(
        func=get_weather,
        name="travel_weather",
        description=(
            "Get weather information for a travel destination. Use this when "
            "weather should influence itinerary activities, packing, or timing."
        ),
    )


travel_knowledge_search_tool = create_travel_knowledge_search_tool()
weather_tool = create_weather_tool()