import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anyio
import httpx
from mcp import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from app.services.mcp_client import (
    PROJECT_ROOT,
    WEATHER_SERVER_MODULE,
    MCPClient,
    MCPClientError,
    new_event_loop,
    weather_server_parameters,
)
from mcp_servers import weather_server

FAKE_SERVER_SCRIPT = Path(__file__).with_name("fake_weather_mcp_server.py")

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


def fake_open_meteo(geocoding=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host.startswith("geocoding"):
            return httpx.Response(200, json=GEOCODING_PAYLOAD if geocoding is None else geocoding)
        return httpx.Response(200, json=FORECAST_PAYLOAD)

    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


class Echo(BaseModel):
    location: str


def recording_server(calls):
    server = MCPServer(name="fake-weather")

    @server.tool(name="get_forecast")
    async def get_forecast(location: str) -> Echo:
        calls.append(location)
        return Echo(location=location)

    @server.tool(name="slow_tool")
    async def slow_tool() -> Echo:
        await anyio.sleep(30)
        return Echo(location="never")

    @server.tool(name="text_tool", structured_output=False)
    async def text_tool() -> str:
        return "plain text only"

    return server


def process_is_running(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class InProcessMCPClientTests(unittest.TestCase):
    def test_get_forecast_through_weather_server(self):
        with patch.object(weather_server, "create_http_client", fake_open_meteo()):
            result = MCPClient(server=weather_server.server).call_tool("get_forecast", {"location": "Paris"})

        self.assertEqual(result["location"], "Paris, Île-de-France, France")
        self.assertEqual(result["timezone"], "Europe/Paris")
        self.assertEqual(result["daily"][0]["conditions"], "clear sky")
        self.assertIn("summary", result)

    def test_tool_name_and_arguments_are_sent(self):
        calls = []

        result = MCPClient(server=recording_server(calls)).call_tool("get_forecast", {"location": "Paris"})

        self.assertEqual(calls, ["Paris"])
        self.assertEqual(result, {"location": "Paris"})

    def test_tool_error_becomes_client_error(self):
        with patch.object(weather_server, "create_http_client", fake_open_meteo(geocoding={})):
            with self.assertRaises(MCPClientError) as context:
                MCPClient(server=weather_server.server).call_tool("get_forecast", {"location": "Atlantis"})

        self.assertEqual(str(context.exception), "MCP tool 'get_forecast' failed: Location not found: Atlantis")

    def test_unknown_tool_becomes_client_error(self):
        with self.assertRaises(MCPClientError) as context:
            MCPClient(server=recording_server([])).call_tool("book_hotel", {})

        self.assertIn("book_hotel", str(context.exception))

    def test_unstructured_result_is_rejected(self):
        with self.assertRaises(MCPClientError) as context:
            MCPClient(server=recording_server([])).call_tool("text_tool", {})

        self.assertEqual(str(context.exception), "MCP tool 'text_tool' returned an unexpected result")

    def test_non_dict_structured_result_is_rejected(self):
        result = SimpleNamespace(is_error=False, structured_content=["unexpected"], content=[])

        with self.assertRaises(MCPClientError):
            MCPClient._parse_result("get_forecast", result)

    def test_long_error_details_are_truncated(self):
        result = SimpleNamespace(is_error=True, content=[SimpleNamespace(text="x" * 5000)])

        with self.assertRaises(MCPClientError) as context:
            MCPClient._parse_result("get_forecast", result)

        self.assertLess(len(str(context.exception)), 400)

    def test_timeout_becomes_client_error(self):
        started = time.monotonic()

        with self.assertRaises(MCPClientError) as context:
            MCPClient(server=recording_server([]), timeout_seconds=0.5).call_tool("slow_tool", {})

        self.assertIn("timed out after 0.5 seconds", str(context.exception))
        self.assertLess(time.monotonic() - started, 5)

    def test_invalid_call_arguments_are_rejected(self):
        client = MCPClient(server=recording_server([]))

        with self.assertRaises(ValueError):
            client.call_tool("", {})
        with self.assertRaises(ValueError):
            client.call_tool("get_forecast", ["Paris"])

    def test_sync_call_inside_running_event_loop_is_rejected(self):
        async def call_from_loop():
            MCPClient(server=recording_server([])).call_tool("get_forecast", {"location": "Paris"})

        with self.assertRaises(MCPClientError) as context:
            asyncio.run(call_from_loop())

        self.assertIn("acall_tool", str(context.exception))

    def test_async_interface_works_inside_event_loop(self):
        calls = []

        result = asyncio.run(
            MCPClient(server=recording_server(calls)).acall_tool("get_forecast", {"location": "Rome"})
        )

        self.assertEqual(result, {"location": "Rome"})

    def test_default_timeout_comes_from_settings(self):
        with patch("app.services.mcp_client.settings", SimpleNamespace(MCP_TIMEOUT_SECONDS=7)):
            self.assertEqual(MCPClient().timeout_seconds, 7)


class StdioConfigurationTests(unittest.TestCase):
    def test_weather_server_is_launched_with_current_interpreter(self):
        parameters = weather_server_parameters()

        self.assertEqual(parameters.command, sys.executable)
        self.assertEqual(parameters.args, ["-m", WEATHER_SERVER_MODULE])
        self.assertEqual(Path(parameters.cwd), PROJECT_ROOT)
        self.assertTrue((PROJECT_ROOT / "mcp_servers" / "weather_server.py").is_file())

    def test_default_client_uses_weather_server_parameters(self):
        self.assertEqual(MCPClient().server, weather_server_parameters())

    @unittest.skipUnless(sys.platform == "win32", "Windows-specific event loop")
    def test_windows_uses_proactor_event_loop_for_subprocesses(self):
        loop = new_event_loop()
        try:
            self.assertIsInstance(loop, asyncio.ProactorEventLoop)
        finally:
            loop.close()


class StdioMCPClientTests(unittest.TestCase):
    def setUp(self):
        handle, self.pid_file = tempfile.mkstemp(suffix=".pid")
        os.close(handle)
        self.addCleanup(os.remove, self.pid_file)

    def fake_server(self, mode):
        return StdioServerParameters(
            command=sys.executable,
            args=[str(FAKE_SERVER_SCRIPT), mode],
            cwd=str(PROJECT_ROOT),
            env={"PYTHONPATH": str(PROJECT_ROOT), "MCP_TEST_PID_FILE": self.pid_file},
        )

    def assert_server_process_stopped(self):
        pid = int(Path(self.pid_file).read_text())
        deadline = time.monotonic() + 3
        while process_is_running(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_is_running(pid), f"MCP server process {pid} was left running")

    def test_stdio_get_forecast_succeeds_and_cleans_up(self):
        result = MCPClient(server=self.fake_server("ok")).call_tool("get_forecast", {"location": "Paris"})

        self.assertEqual(result["location"], "Paris, Île-de-France, France")
        self.assertEqual(len(result["daily"]), 2)
        self.assert_server_process_stopped()

    def test_real_weather_server_module_reports_tool_errors_without_network(self):
        with self.assertRaises(MCPClientError) as context:
            MCPClient().call_tool("get_forecast", {"location": "   "})

        self.assertEqual(
            str(context.exception), "MCP tool 'get_forecast' failed: location must be a non-empty string"
        )

    def test_missing_server_executable_is_startup_failure(self):
        server = StdioServerParameters(command="definitely-not-an-mcp-server-xyz", cwd=str(PROJECT_ROOT))

        with self.assertRaises(MCPClientError) as context:
            MCPClient(server=server).call_tool("get_forecast", {"location": "Paris"})

        self.assertEqual(str(context.exception), "MCP server process could not be started")

    def test_server_that_exits_on_startup_is_reported(self):
        with self.assertRaises(MCPClientError) as context:
            MCPClient(server=self.fake_server("crash")).call_tool("get_forecast", {"location": "Paris"})

        self.assertEqual(str(context.exception), "MCP server closed the connection unexpectedly")
        self.assert_server_process_stopped()

    def test_non_mcp_process_is_connection_failure(self):
        server = StdioServerParameters(command=sys.executable, args=["-c", "print('not an MCP server')"])

        with self.assertRaises(MCPClientError) as context:
            MCPClient(server=server).call_tool("get_forecast", {"location": "Paris"})

        self.assertEqual(str(context.exception), "MCP server closed the connection unexpectedly")

    def test_server_process_dying_mid_call_is_reported_and_cleaned_up(self):
        with self.assertRaises(MCPClientError) as context:
            MCPClient(server=self.fake_server("die")).call_tool("get_forecast", {"location": "Paris"})

        self.assertEqual(str(context.exception), "MCP server closed the connection unexpectedly")
        self.assert_server_process_stopped()

    def test_stdio_timeout_stops_server_process(self):
        started = time.monotonic()

        with self.assertRaises(MCPClientError) as context:
            MCPClient(server=self.fake_server("slow"), timeout_seconds=3).call_tool(
                "get_forecast", {"location": "Paris"}
            )

        self.assertIn("timed out after 3 seconds", str(context.exception))
        self.assertLess(time.monotonic() - started, 10)
        self.assert_server_process_stopped()

    @unittest.skipUnless(sys.platform == "win32", "Windows-specific event loop policy")
    def test_stdio_works_when_selector_policy_is_active(self):
        previous = asyncio.get_event_loop_policy()
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        try:
            result = MCPClient(server=self.fake_server("ok")).call_tool("get_forecast", {"location": "Paris"})
        finally:
            asyncio.set_event_loop_policy(previous)

        self.assertEqual(result["timezone"], "Europe/Paris")
        self.assert_server_process_stopped()


if __name__ == "__main__":
    unittest.main()
