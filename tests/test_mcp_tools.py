import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.services.agent_service import OrchestrationError, TravelPlanningOrchestrator
from app.services.ai_service import GeneratedItinerary
from app.services.mcp_client import MCPClient, MCPClientError
from app.tools import default_travel_tools, travel_knowledge_search_tool, weather_tool
from app.tools.mcp_tools import MCPWeatherService, create_mcp_weather_tool
from app.tools.travel_tools import create_weather_tool
from mcp_servers import weather_server

MCP_FORECAST = {
    "location": "Paris, Île-de-France, France",
    "latitude": 48.85341,
    "longitude": 2.3488,
    "timezone": "Europe/Paris",
    "summary": "1-day forecast for Paris, Île-de-France, France: highs 21 to 21°C, lows 12 to 12°C, mostly clear sky.",
    "daily": [
        {
            "date": "2026-10-03",
            "conditions": "clear sky",
            "temperature_max_c": 21.0,
            "temperature_min_c": 12.0,
            "precipitation_mm": 0.0,
            "precipitation_probability_percent": 5,
        }
    ],
}

GEOCODING_PAYLOAD = {
    "results": [
        {"name": "Paris", "latitude": 48.85341, "longitude": 2.3488, "country": "France", "admin1": "Île-de-France"}
    ]
}
FORECAST_PAYLOAD = {
    "timezone": "Europe/Paris",
    "daily": {
        "time": ["2026-10-03"],
        "weather_code": [0],
        "temperature_2m_max": [21.0],
        "temperature_2m_min": [12.0],
        "precipitation_sum": [0.0],
        "precipitation_probability_max": [5],
    },
}


class FakeMCPClient:
    def __init__(self, result=None, error=None):
        self.result = MCP_FORECAST if result is None else result
        self.error = error
        self.calls = []

    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        if self.error:
            raise self.error
        return self.result


def fake_open_meteo(geocoding=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host.startswith("geocoding"):
            return httpx.Response(200, json=GEOCODING_PAYLOAD if geocoding is None else geocoding)
        return httpx.Response(200, json=FORECAST_PAYLOAD)

    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


def sample_itinerary():
    return GeneratedItinerary.model_validate(
        {
            "destination": "Paris",
            "days": [
                {
                    "day": 1,
                    "activities": [
                        {"time": "morning", "activity": "Louvre", "location": "Paris", "estimated_cost": "$25"}
                    ],
                }
            ],
            "total_estimated_cost": "$25",
        }
    )


class FakeModel:
    def __init__(self, planner_responses):
        self.planner_responses = iter(planner_responses)
        self.planner_inputs = []
        self.finalization_inputs = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return FakePlanner(self)

    def with_structured_output(self, schema):
        return FakeFinalizer(self)


class FakePlanner:
    def __init__(self, model):
        self.model = model

    def invoke(self, messages):
        self.model.planner_inputs.append(messages)
        return next(self.model.planner_responses)


class FakeFinalizer:
    def __init__(self, model):
        self.model = model

    def invoke(self, messages):
        self.model.finalization_inputs.append(messages)
        return sample_itinerary()


def weather_call(location="Paris"):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "travel_weather", "args": {"location": location}, "id": "weather-call", "type": "tool_call"}
        ],
    )


class MCPWeatherToolTests(unittest.TestCase):
    def test_tool_keeps_existing_name_and_argument_schema(self):
        tool = create_mcp_weather_tool(FakeMCPClient())

        self.assertEqual(tool.name, "travel_weather")
        self.assertEqual(tool.args, create_weather_tool().args)
        self.assertEqual(list(tool.args), ["location"])
        self.assertEqual(tool.description, weather_tool.description)

    def test_tool_calls_get_forecast_with_location(self):
        client = FakeMCPClient()

        create_mcp_weather_tool(client).invoke({"location": "  Paris "})

        self.assertEqual(client.calls, [("get_forecast", {"location": "Paris"})])

    def test_structured_mcp_result_is_returned_in_existing_shape(self):
        result = create_mcp_weather_tool(FakeMCPClient()).invoke({"location": "Paris"})

        self.assertEqual(result, {"location": "Paris", "weather": MCP_FORECAST})
        self.assertIn("summary", result["weather"])

    def test_empty_location_is_rejected_before_mcp_call(self):
        client = FakeMCPClient()

        with self.assertRaises(ValueError):
            create_mcp_weather_tool(client).invoke({"location": "  "})

        self.assertEqual(client.calls, [])

    def test_mcp_client_error_becomes_clean_tool_error(self):
        client = FakeMCPClient(error=MCPClientError("MCP tool 'get_forecast' failed: Location not found: Atlantis"))

        with self.assertRaises(RuntimeError) as context:
            create_mcp_weather_tool(client).invoke({"location": "Atlantis"})

        self.assertNotIsInstance(context.exception, MCPClientError)
        self.assertEqual(
            str(context.exception),
            "Weather lookup failed: MCP tool 'get_forecast' failed: Location not found: Atlantis",
        )

    def test_weather_service_defaults_to_mcp_client(self):
        self.assertIsInstance(MCPWeatherService().client, MCPClient)


class DefaultTravelToolsTests(unittest.TestCase):
    def tools_with_flag(self, enabled):
        with patch.object(settings, "MCP_WEATHER_ENABLED", enabled):
            return default_travel_tools()

    def test_mcp_disabled_uses_local_weather_tool(self):
        with patch("app.tools.mcp_tools.MCPClient", side_effect=AssertionError("MCP must not be used")):
            tools = self.tools_with_flag(False)
            result = tools[1].invoke({"location": "Paris"})

        self.assertIs(tools[1], weather_tool)
        self.assertEqual(result, {"location": "Paris", "weather": {"summary": "mostly sunny, average 24°C, light breeze"}})

    def test_mcp_enabled_uses_mcp_weather_tool(self):
        client = FakeMCPClient()

        with patch("app.tools.mcp_tools.MCPClient", return_value=client):
            tools = self.tools_with_flag(True)
            result = tools[1].invoke({"location": "Paris"})

        self.assertIsNot(tools[1], weather_tool)
        self.assertEqual(client.calls, [("get_forecast", {"location": "Paris"})])
        self.assertEqual(result["weather"], MCP_FORECAST)

    def test_exactly_one_weather_tool_in_both_modes(self):
        for enabled in (False, True):
            names = [tool.name for tool in self.tools_with_flag(enabled)]
            self.assertEqual(names, ["travel_knowledge_search", "travel_weather"])

    def test_knowledge_tool_is_unchanged_in_both_modes(self):
        for enabled in (False, True):
            self.assertIs(self.tools_with_flag(enabled)[0], travel_knowledge_search_tool)

    def test_orchestrator_defaults_follow_flag(self):
        for enabled in (False, True):
            with patch.object(settings, "MCP_WEATHER_ENABLED", enabled):
                model = FakeModel([])
                orchestrator = TravelPlanningOrchestrator(model=model)

            self.assertEqual([tool.name for tool in model.bound_tools], ["travel_knowledge_search", "travel_weather"])
            self.assertEqual(orchestrator.tools[1] is weather_tool, not enabled)


class MCPBoundaryIntegrationTests(unittest.TestCase):
    """travel_weather -> MCPWeatherService -> MCPClient -> in-process weather MCP server, with Open-Meteo faked."""

    def in_process_tool(self, geocoding=None):
        patcher = patch.object(weather_server, "create_http_client", fake_open_meteo(geocoding))
        patcher.start()
        self.addCleanup(patcher.stop)
        return create_mcp_weather_tool(MCPClient(server=weather_server.server))

    def test_langchain_tool_reaches_mcp_server(self):
        result = self.in_process_tool().invoke({"location": "Paris"})

        self.assertEqual(result["location"], "Paris")
        self.assertEqual(result["weather"]["location"], "Paris, Île-de-France, France")
        self.assertEqual(result["weather"]["timezone"], "Europe/Paris")
        self.assertEqual(result["weather"]["daily"][0]["conditions"], "clear sky")

    def test_mcp_server_tool_error_reaches_langchain_tool(self):
        tool = self.in_process_tool(geocoding={})

        with self.assertRaises(RuntimeError) as context:
            tool.invoke({"location": "Atlantis"})

        self.assertEqual(
            str(context.exception),
            "Weather lookup failed: MCP tool 'get_forecast' failed: Location not found: Atlantis",
        )

    def test_langgraph_runs_mcp_backed_weather_tool(self):
        model = FakeModel([weather_call("Paris"), AIMessage(content="Plan around the forecast.")])

        result = TravelPlanningOrchestrator(model=model, tools=[self.in_process_tool()]).invoke(
            "Plan a day in Paris around the weather."
        )

        self.assertEqual(result.destination, "Paris")
        self.assertIn("Paris, Île-de-France, France", str(model.planner_inputs[1]))
        self.assertIn("clear sky", str(model.finalization_inputs[0]))

    def test_langgraph_wraps_mcp_failures_in_orchestration_error(self):
        model = FakeModel([weather_call("Atlantis")])

        with self.assertRaises(OrchestrationError) as context:
            TravelPlanningOrchestrator(model=model, tools=[self.in_process_tool(geocoding={})]).invoke(
                "Plan a day in Atlantis."
            )

        self.assertIn("Weather lookup failed", str(context.exception))
        self.assertIn("Location not found: Atlantis", str(context.exception))


if __name__ == "__main__":
    unittest.main()
