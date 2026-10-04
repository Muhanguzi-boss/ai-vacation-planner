"""Phase 6 MCP weather server.

Exposes a single MCP tool, ``get_forecast(location)``, which geocodes a place name and
returns a daily forecast using the free, keyless Open-Meteo geocoding and forecast APIs.
It communicates over the Model Context Protocol (stdio transport by default) and is
launched with ``python -m mcp_servers.weather_server``.

This is a data server only: it contains no LLM logic, needs no API keys, and has no
dependency on the FastAPI application.
"""

from collections import Counter
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ValidationError

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT_SECONDS = 10.0
FORECAST_DAYS = 7
GEOCODING_CANDIDATES = 10
RAIN_DAY_THRESHOLD_MM = 1.0
DAILY_FIELDS = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
)

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


class DailyForecast(BaseModel):
    date: str
    conditions: str
    temperature_max_c: float
    temperature_min_c: float
    precipitation_mm: float | None
    precipitation_probability_percent: int | None


class WeatherForecast(BaseModel):
    location: str
    latitude: float
    longitude: float
    timezone: str
    summary: str
    daily: list[DailyForecast]


class _Place(BaseModel):
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    country_code: str | None = None
    admin1: str | None = None

    @property
    def label(self) -> str:
        return ", ".join(dict.fromkeys(part for part in (self.name, self.admin1, self.country) if part))


def create_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)


async def _get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any], service: str) -> Any:
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        raise ToolError(f"{service} service timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise ToolError(f"{service} service returned HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise ToolError(f"{service} service is unreachable") from exc
    except ValueError as exc:
        raise ToolError(f"{service} service returned an unexpected response") from exc


async def geocode(client: httpx.AsyncClient, location: str) -> _Place:
    name, *qualifiers = [part.strip() for part in location.split(",") if part.strip()]
    payload = await _get_json(
        client,
        GEOCODING_URL,
        {"name": name, "count": GEOCODING_CANDIDATES, "language": "en", "format": "json"},
        "Geocoding",
    )
    if not isinstance(payload, dict):
        raise ToolError("Geocoding service returned an unexpected response")
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise ToolError("Geocoding service returned an unexpected response")
    try:
        places = [_Place.model_validate(result) for result in results]
    except ValidationError as exc:
        raise ToolError("Geocoding service returned an unexpected response") from exc

    wanted = [qualifier.lower() for qualifier in qualifiers]
    for place in places:
        context = {(value or "").lower() for value in (place.country, place.country_code, place.admin1)}
        if all(qualifier in context for qualifier in wanted):
            return place
    raise ToolError(f"Location not found: {location}")


def _summarize(location: str, days: list[DailyForecast]) -> str:
    highs = [day.temperature_max_c for day in days]
    lows = [day.temperature_min_c for day in days]
    conditions = Counter(day.conditions for day in days).most_common(1)[0][0]
    rain_days = sum(1 for day in days if (day.precipitation_mm or 0) >= RAIN_DAY_THRESHOLD_MM)
    return (
        f"{len(days)}-day forecast for {location}: highs {min(highs):.0f} to {max(highs):.0f}°C, "
        f"lows {min(lows):.0f} to {max(lows):.0f}°C, mostly {conditions}, "
        f"{rain_days} day(s) with {RAIN_DAY_THRESHOLD_MM:.0f} mm or more of precipitation."
    )


async def fetch_forecast(client: httpx.AsyncClient, location: str) -> WeatherForecast:
    if not isinstance(location, str) or not location.strip(" ,"):
        raise ToolError("location must be a non-empty string")
    place = await geocode(client, location.strip())

    payload = await _get_json(
        client,
        FORECAST_URL,
        {
            "latitude": place.latitude,
            "longitude": place.longitude,
            "daily": ",".join(DAILY_FIELDS),
            "timezone": "auto",
            "forecast_days": FORECAST_DAYS,
        },
        "Forecast",
    )
    try:
        daily = payload["daily"]
        dates = daily["time"]
        columns = [daily[field] for field in DAILY_FIELDS]
        if not dates or any(len(column) != len(dates) for column in columns):
            raise ValueError("daily columns have inconsistent lengths")
        days = [
            DailyForecast(
                date=date,
                conditions=WEATHER_CODES.get(code, "unknown"),
                temperature_max_c=high,
                temperature_min_c=low,
                precipitation_mm=precipitation,
                precipitation_probability_percent=probability,
            )
            for date, code, high, low, precipitation, probability in zip(dates, *columns)
        ]
        timezone = str(payload["timezone"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError("Forecast service returned an unexpected response") from exc

    return WeatherForecast(
        location=place.label,
        latitude=place.latitude,
        longitude=place.longitude,
        timezone=timezone,
        summary=_summarize(place.label, days),
        daily=days,
    )


server = MCPServer(
    name="travel-weather",
    instructions="Weather forecasts for travel destinations, backed by Open-Meteo.",
)


@server.tool(
    name="get_forecast",
    description=(
        "Get a daily weather forecast for a travel destination. Accepts a place name such as "
        "'Paris' or 'Paris, France' and returns the resolved location, coordinates, timezone, "
        f"a one-line summary, and up to {FORECAST_DAYS} days of conditions, temperatures, and precipitation."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
)
async def get_forecast(location: str) -> WeatherForecast:
    async with create_http_client() as client:
        return await fetch_forecast(client, location)


if __name__ == "__main__":
    server.run()
