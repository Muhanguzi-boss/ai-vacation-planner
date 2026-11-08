"""Runs the real weather MCP server over stdio with Open-Meteo replaced by canned responses.

Used by tests/test_mcp_client.py so stdio tests never touch the network.
Usage: python tests/fake_weather_mcp_server.py <ok|slow|crash|die>
  ok     serve normally
  slow   Open-Meteo requests never finish
  crash  exit before serving
  die    exit abruptly while handling a tool call
The process writes its PID to the file named by MCP_TEST_PID_FILE when that variable is set.
"""

import asyncio
import os
import sys

import httpx

from mcp_servers import weather_server

GEOCODING_PAYLOAD = {
    "results": [
        {"name": "Paris", "latitude": 48.85341, "longitude": 2.3488, "country": "France", "admin1": "Île-de-France"}
    ]
}
FORECAST_PAYLOAD = {
    "timezone": "Europe/Paris",
    "daily": {
        "time": ["2026-10-03", "2026-10-04"],
        "weather_code": [3, 61],
        "temperature_2m_max": [18.4, 16.2],
        "temperature_2m_min": [10.1, 11.7],
        "precipitation_sum": [0.0, 4.2],
        "precipitation_probability_max": [10, 80],
    },
}

mode = sys.argv[1] if len(sys.argv) > 1 else "ok"


async def open_meteo(request: httpx.Request) -> httpx.Response:
    if mode == "slow":
        await asyncio.sleep(600)
    if mode == "die":
        os._exit(3)
    if request.url.host.startswith("geocoding"):
        return httpx.Response(200, json=GEOCODING_PAYLOAD)
    return httpx.Response(200, json=FORECAST_PAYLOAD)


if __name__ == "__main__":
    if pid_file := os.environ.get("MCP_TEST_PID_FILE"):
        with open(pid_file, "w") as handle:
            handle.write(str(os.getpid()))
    if mode == "crash":
        sys.exit(1)
    weather_server.create_http_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(open_meteo))
    weather_server.server.run()
