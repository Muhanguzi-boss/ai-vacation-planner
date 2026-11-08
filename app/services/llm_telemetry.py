"""LangChain callback handlers that turn model and tool runs into telemetry events.

The handlers read only run identifiers, timing, the model name, token counts, the tool name
and the exception *type*. Prompts, messages, completions, tool inputs/outputs and exception
messages are never read or recorded.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from app.core.telemetry import (
    FAILURE,
    LLM_REQUEST,
    NO_ERROR,
    SUCCESS,
    TOOL_EXECUTION,
    Telemetry,
    elapsed_ms,
    exception_category,
    telemetry,
)

_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


class _RunTimer:
    def __init__(self) -> None:
        self._runs: dict[UUID, tuple[float, dict[str, str]]] = {}
        self._lock = threading.Lock()

    def start(self, run_id: UUID, tags: dict[str, str]) -> None:
        with self._lock:
            self._runs[run_id] = (time.perf_counter(), tags)

    def stop(self, run_id: UUID) -> tuple[float | None, dict[str, str]]:
        with self._lock:
            started, tags = self._runs.pop(run_id, (None, {}))
        return (elapsed_ms(started) if started is not None else None), tags


def _model_from_invocation(kwargs: dict[str, Any]) -> str | None:
    params = kwargs.get("invocation_params") or {}
    for key in ("model", "model_name"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _usage_and_model(response: Any) -> tuple[dict[str, int], str | None]:
    """Extract token usage and the served model name from an ``LLMResult``."""
    usage: dict[str, int] = {}
    model: str | None = None
    for generations in getattr(response, "generations", None) or []:
        for generation in generations:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            metadata = getattr(message, "usage_metadata", None) or {}
            for key in _TOKEN_FIELDS:
                value = metadata.get(key)
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
            response_metadata = getattr(message, "response_metadata", None) or {}
            model = model or response_metadata.get("model_name") or response_metadata.get("model")

    llm_output = getattr(response, "llm_output", None) or {}
    if not usage and isinstance(llm_output.get("usage"), dict):
        raw = llm_output["usage"]
        for key in ("input_tokens", "output_tokens"):
            if isinstance(raw.get(key), int):
                usage[key] = raw[key]
        if usage:
            usage["total_tokens"] = sum(usage.values())
    model = model or llm_output.get("model_name") or llm_output.get("model")
    return usage, model if isinstance(model, str) else None


class LLMTelemetryCallback(BaseCallbackHandler):
    """Records latency, outcome, model and token usage for every chat-model call.

    Attach it to the chat model itself (``callbacks=[...]``) so planner, finalizer and
    vision calls are all covered without changing how those calls are made.
    """

    ignore_retriever = True
    ignore_chain = True
    ignore_agent = True

    def __init__(self, telemetry_client: Telemetry | None = None) -> None:
        self.telemetry = telemetry_client or telemetry
        self._timer = _RunTimer()

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self._start(run_id, metadata, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs) -> None:
        self._start(run_id, metadata, kwargs)

    def _start(self, run_id: UUID, metadata: dict[str, Any] | None, kwargs: dict[str, Any]) -> None:
        node = (metadata or {}).get("langgraph_node")
        self._timer.start(
            run_id,
            {
                "model": _model_from_invocation(kwargs) or "unknown",
                "node": node if isinstance(node, str) else "standalone",
            },
        )

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        duration, tags = self._timer.stop(run_id)
        usage, served_model = _usage_and_model(response)
        measurements: dict[str, float] = dict(usage)
        if duration is not None:
            measurements["duration_ms"] = duration
        self.telemetry.emit(
            LLM_REQUEST,
            tags={
                **tags,
                "model": served_model or tags.get("model", "unknown"),
                "outcome": SUCCESS,
                "error_category": NO_ERROR,
            },
            measurements=measurements,
        )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        duration, tags = self._timer.stop(run_id)
        self.telemetry.emit(
            LLM_REQUEST,
            tags={**tags, "outcome": FAILURE, "error_category": exception_category(error)},
            measurements={"duration_ms": duration} if duration is not None else {},
        )


class ToolTelemetryCallback(BaseCallbackHandler):
    """Records latency, outcome and error category for each tool execution.

    Create one per graph invocation (pass it in the graph ``config``); ``failed`` then
    tells the caller whether any tool failed during that invocation.
    """

    # Tool events are dispatched under LangChain's ``ignore_agent`` flag, so it stays False.
    ignore_llm = True
    ignore_chat_model = True
    ignore_retriever = True
    ignore_chain = True

    def __init__(self, telemetry_client: Telemetry | None = None) -> None:
        self.telemetry = telemetry_client or telemetry
        self._timer = _RunTimer()
        self.failed = False

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._timer.start(run_id, {"tool": name if isinstance(name, str) else "unknown"})

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        self._record(run_id, SUCCESS, NO_ERROR)

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        self.failed = True
        self._record(run_id, FAILURE, exception_category(error))

    def _record(self, run_id: UUID, outcome: str, category: str) -> None:
        duration, tags = self._timer.stop(run_id)
        self.telemetry.emit(
            TOOL_EXECUTION,
            tags={**tags, "outcome": outcome, "error_category": category},
            measurements={"duration_ms": duration} if duration is not None else {},
        )


llm_telemetry_callback = LLMTelemetryCallback()
