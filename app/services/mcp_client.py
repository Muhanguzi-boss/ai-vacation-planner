"""Application-side MCP client.

Starts the local weather MCP server (``python -m mcp_servers.weather_server``) with the
current interpreter and talks to it over the MCP stdio transport. The MCP SDK is async;
``MCPClient.call_tool`` runs each call on a private event loop so the synchronous FastAPI
routes and LangGraph orchestration can use it unchanged. Each call starts a server process
and shuts it down before returning.

This module contains no LLM logic.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, MCPError, StdioServerParameters
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT

from app.core.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_SERVER_MODULE = "mcp_servers.weather_server"
MAX_ERROR_DETAIL_LENGTH = 300


class MCPClientError(RuntimeError):
    """Raised when an MCP server cannot be reached or an MCP tool call fails."""


def weather_server_parameters() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", WEATHER_SERVER_MODULE],
        cwd=str(PROJECT_ROOT),
    )


def new_event_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    return asyncio.new_event_loop()


def _root_cause(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


def _shorten(text: str) -> str:
    return text if len(text) <= MAX_ERROR_DETAIL_LENGTH else text[:MAX_ERROR_DETAIL_LENGTH] + "..."


class MCPClient:
    def __init__(self, server: Any | None = None, timeout_seconds: float | None = None) -> None:
        self.server = server or weather_server_parameters()
        self.timeout_seconds = timeout_seconds or settings.MCP_TIMEOUT_SECONDS

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise MCPClientError("MCPClient.call_tool cannot run inside an event loop; await acall_tool instead")

        with asyncio.Runner(loop_factory=new_event_loop) as runner:
            return runner.run(self.acall_tool(tool_name, arguments))

    async def acall_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a dictionary")

        try:
            with anyio.fail_after(self.timeout_seconds):
                async with Client(self.server, read_timeout_seconds=self.timeout_seconds) as client:
                    result = await client.call_tool(tool_name, arguments)
        except Exception as exc:
            raise self._translate_error(tool_name, exc) from exc
        return self._parse_result(tool_name, result)

    def _translate_error(self, tool_name: str, exc: Exception) -> MCPClientError:
        cause = _root_cause(exc)
        if isinstance(cause, TimeoutError) or (isinstance(cause, MCPError) and cause.code == REQUEST_TIMEOUT):
            return MCPClientError(f"MCP tool '{tool_name}' timed out after {self.timeout_seconds:g} seconds")
        if isinstance(cause, MCPError) and cause.code == CONNECTION_CLOSED:
            return MCPClientError("MCP server closed the connection unexpectedly")
        if isinstance(cause, MCPError):
            return MCPClientError(f"MCP request failed: {_shorten(cause.message)}")
        if isinstance(cause, OSError):
            return MCPClientError("MCP server process could not be started")
        return MCPClientError(f"MCP tool '{tool_name}' failed unexpectedly ({type(cause).__name__})")

    @staticmethod
    def _parse_result(tool_name: str, result: Any) -> dict[str, Any]:
        if result.is_error:
            detail = " ".join(
                block.text for block in result.content if isinstance(getattr(block, "text", None), str)
            ).strip()
            detail = detail.removeprefix(f"Error executing tool {tool_name}: ")
            raise MCPClientError(f"MCP tool '{tool_name}' failed: {_shorten(detail) or 'no details provided'}")
        if not isinstance(result.structured_content, dict):
            raise MCPClientError(f"MCP tool '{tool_name}' returned an unexpected result")
        return result.structured_content
