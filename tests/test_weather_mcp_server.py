import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError

from mcp_servers import weather_server
from mcp_servers.weather_server import (
    DAILY_FIELDS,
    FORECAST_URL,
    GEOCODING_URL,
    WeatherForecast,
    fetch_forecast,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARIS_FRANCE = {
    "name": "Paris",
    "latitude": 48.85341,
    "longitude": 2.3488,
    "country": "France",
    "country_code": "FR",
    "admin1": "Île-de-France",
    "timezone": "Europe/Paris",
}
PARIS_TEXAS = {
    "name": "Paris",
    "latitude": 33.66094,
    "longitude": -95.55551,
    "country": "United States",
    "country_code": "US",
    "admin1": "Texas",
    "timezone": "America/Chicago",
}
GEOCODING_PAYLOAD = {"results": [PARIS_FRANCE, PARIS_TEXAS], "generationtime_ms": 0.5}
FORECAST_PAYLOAD = {
    "latitude": 48.86,
    "longitude": 2.3399997,
    "timezone": "Europe/Paris",
    "daily_units": {"temperature_2m_max": "°C"},
    "daily": {
        "time": ["2026-10-03", "2026-10-04", "2026-10-05"],
        "weather_code": [3, 61, 3],
        "temperature_2m_max": [18.4, 16.2, 19.6],
        "temperature_2m_min": [10.1, 11.7, 9.3],
        "precipitation_sum": [0.0, 4.2, 0.3],
        "precipitation_probability_max": [10, 80, None],
    },
}


class FakeOpenMeteo:
    def __init__(self, geocoding=None, forecast=None):
        self.geocoding = geocoding or httpx.Response(200, json=GEOCODING_PAYLOAD)
        self.forecast = forecast or httpx.Response(200, json=FORECAST_PAYLOAD)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url.copy_with(query=None))
        handler = {GEOCODING_URL: self.geocoding, FORECAST_URL: self.forecast}[url]
        if isinstance(handler, Exception):
            raise handler
        return handler

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


class FetchForecastTests(unittest.IsolatedAsyncioTestCase):
    async def forecast_for(self, location, api=None):
        api = api or FakeOpenMeteo()
        async with api.client() as client:
            return await fetch_forecast(client, location)

    async def assert_tool_error(self, location, api, message):
        with self.assertRaises(ToolError) as context:
            await self.forecast_for(location, api)
        self.assertIn(message, str(context.exception))
        return context.exception

    async def test_valid_location_returns_structured_forecast(self):
        result = await self.forecast_for("Paris")

        self.assertIsInstance(result, WeatherForecast)
        self.assertEqual(result.location, "Paris, Île-de-France, France")
        self.assertEqual((result.latitude, result.longitude), (48.85341, 2.3488))
        self.assertEqual(result.timezone, "Europe/Paris")
        self.assertEqual(len(result.daily), 3)
        self.assertEqual(
            result.daily[1].model_dump(),
            {
                "date": "2026-10-04",
                "conditions": "light rain",
                "temperature_max_c": 16.2,
                "temperature_min_c": 11.7,
                "precipitation_mm": 4.2,
                "precipitation_probability_percent": 80,
            },
        )
        self.assertIsNone(result.daily[2].precipitation_probability_percent)
        self.assertEqual(
            result.summary,
            "3-day forecast for Paris, Île-de-France, France: highs 16 to 20°C, "
            "lows 9 to 12°C, mostly overcast, 1 day(s) with 1 mm or more of precipitation.",
        )

    async def test_open_meteo_is_queried_with_expected_parameters(self):
        api = FakeOpenMeteo()

        await self.forecast_for("Paris", api)

        geocoding, forecast = api.requests
        self.assertEqual(geocoding.url.params["name"], "Paris")
        self.assertEqual(forecast.url.params["latitude"], "48.85341")
        self.assertEqual(forecast.url.params["longitude"], "2.3488")
        self.assertEqual(forecast.url.params["daily"], ",".join(DAILY_FIELDS))
        self.assertEqual(forecast.url.params["timezone"], "auto")

    async def test_qualifiers_select_matching_place(self):
        api = FakeOpenMeteo()

        result = await self.forecast_for("Paris, Texas", api)

        self.assertEqual(result.location, "Paris, Texas, United States")
        self.assertEqual(api.requests[0].url.params["name"], "Paris")
        self.assertEqual(api.requests[1].url.params["latitude"], "33.66094")

    async def test_country_code_qualifier_is_supported(self):
        result = await self.forecast_for("Paris, FR")

        self.assertEqual(result.location, "Paris, Île-de-France, France")

    async def test_repeated_place_names_are_not_duplicated_in_label(self):
        tokyo = {"name": "Tokyo", "latitude": 35.6895, "longitude": 139.69171, "country": "Japan", "admin1": "Tokyo"}
        api = FakeOpenMeteo(geocoding=httpx.Response(200, json={"results": [tokyo]}))

        result = await self.forecast_for("Tokyo", api)

        self.assertEqual(result.location, "Tokyo, Japan")

    async def test_location_not_found(self):
        api = FakeOpenMeteo(geocoding=httpx.Response(200, json={"generationtime_ms": 0.2}))

        await self.assert_tool_error("Atlantis", api, "Location not found: Atlantis")
        self.assertEqual(len(api.requests), 1)

    async def test_unmatched_qualifier_is_not_found(self):
        await self.assert_tool_error("Paris, Japan", FakeOpenMeteo(), "Location not found: Paris, Japan")

    async def test_empty_location_is_rejected_without_http_calls(self):
        for location in ("", "   ", " , "):
            api = FakeOpenMeteo()
            await self.assert_tool_error(location, api, "location must be a non-empty string")
            self.assertEqual(api.requests, [])

    async def test_geocoding_http_failure(self):
        api = FakeOpenMeteo(geocoding=httpx.Response(500, text="<html>internal stack trace</html>"))

        error = await self.assert_tool_error("Paris", api, "Geocoding service returned HTTP 500")
        self.assertNotIn("stack trace", str(error))

    async def test_forecast_http_failure(self):
        api = FakeOpenMeteo(forecast=httpx.Response(503, text="Service Unavailable"))

        await self.assert_tool_error("Paris", api, "Forecast service returned HTTP 503")

    async def test_timeout(self):
        api = FakeOpenMeteo(forecast=httpx.ReadTimeout("timed out"))

        await self.assert_tool_error("Paris", api, "Forecast service timed out")

    async def test_connection_failure(self):
        api = FakeOpenMeteo(geocoding=httpx.ConnectError("dns failure"))

        await self.assert_tool_error("Paris", api, "Geocoding service is unreachable")

    async def test_malformed_geocoding_responses(self):
        for response in (
            httpx.Response(200, text="not json"),
            httpx.Response(200, json=["unexpected"]),
            httpx.Response(200, json={"results": "unexpected"}),
            httpx.Response(200, json={"results": [{"name": "Paris"}]}),
        ):
            api = FakeOpenMeteo(geocoding=response)
            await self.assert_tool_error("Paris", api, "Geocoding service returned an unexpected response")
            self.assertEqual(len(api.requests), 1)

    async def test_malformed_forecast_responses(self):
        daily = FORECAST_PAYLOAD["daily"]
        for payload in (
            {"timezone": "Europe/Paris"},
            {**FORECAST_PAYLOAD, "daily": {**daily, "temperature_2m_max": [18.4]}},
            {**FORECAST_PAYLOAD, "daily": {**daily, "temperature_2m_min": [None, 11.7, 9.3]}},
            {**FORECAST_PAYLOAD, "daily": {**daily, "time": []}},
            {"daily": daily},
            ["unexpected"],
        ):
            api = FakeOpenMeteo(forecast=httpx.Response(200, json=payload))
            await self.assert_tool_error("Paris", api, "Forecast service returned an unexpected response")

    async def test_unknown_weather_code_is_reported_as_unknown(self):
        payload = {**FORECAST_PAYLOAD, "daily": {**FORECAST_PAYLOAD["daily"], "weather_code": [3, 42, None]}}

        result = await self.forecast_for("Paris", FakeOpenMeteo(forecast=httpx.Response(200, json=payload)))

        self.assertEqual([day.conditions for day in result.daily], ["overcast", "unknown", "unknown"])


class WeatherMCPServerTests(unittest.IsolatedAsyncioTestCase):
    def use_api(self, api):
        patcher = patch.object(weather_server, "create_http_client", api.client)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_get_forecast_tool_is_registered(self):
        async with Client(weather_server.server) as client:
            tools = (await client.list_tools()).tools

        self.assertEqual([tool.name for tool in tools], ["get_forecast"])
        [tool] = tools
        self.assertEqual(tool.input_schema["required"], ["location"])
        self.assertEqual(tool.input_schema["properties"]["location"]["type"], "string")
        self.assertEqual(set(tool.output_schema["properties"]), set(WeatherForecast.model_fields))
        self.assertTrue(tool.annotations.read_only_hint)

    async def test_tool_call_returns_structured_content(self):
        self.use_api(FakeOpenMeteo())

        async with Client(weather_server.server) as client:
            result = await client.call_tool("get_forecast", {"location": "Paris"})

        self.assertFalse(result.is_error)
        forecast = WeatherForecast.model_validate(result.structured_content)
        self.assertEqual(forecast.location, "Paris, Île-de-France, France")
        self.assertEqual(len(forecast.daily), 3)

    async def test_tool_errors_are_returned_without_raw_responses(self):
        self.use_api(FakeOpenMeteo(geocoding=httpx.Response(502, text="<html>upstream secret</html>")))

        async with Client(weather_server.server) as client:
            result = await client.call_tool("get_forecast", {"location": "Paris"})

        self.assertTrue(result.is_error)
        text = result.content[0].text
        self.assertIn("Geocoding service returned HTTP 502", text)
        self.assertNotIn("upstream secret", text)

    async def test_missing_location_argument_is_rejected(self):
        async with Client(weather_server.server) as client:
            result = await client.call_tool("get_forecast", {})

        self.assertTrue(result.is_error)

    async def test_server_runs_as_module_over_stdio(self):
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_servers.weather_server"],
            cwd=str(PROJECT_ROOT),
        )

        async with Client(parameters) as client:
            tools = (await client.list_tools()).tools

        self.assertEqual([tool.name for tool in tools], ["get_forecast"])


if __name__ == "__main__":
    unittest.main()
