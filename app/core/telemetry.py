"""Privacy-safe application telemetry.

Every observation is a ``TelemetryEvent``: a dotted event name, a set of low-cardinality
string tags, and numeric measurements. Sinks decide where events go (a structured JSON log
line by default; an in-memory list in tests).

Privacy contract (enforced here, not left to callers):

* Tags are restricted to an allowlist of keys and normalized to short, Datadog-compatible
  values. Unknown tag keys are dropped.
* Callers pass only code-defined values: HTTP method, route *template* (never the raw path or
  query string), status code, model name, tool name, outcome and error category.
* No prompt, transcript, request/response body, tool input/output, user identifier, header,
  credential, or exception message is ever accepted. Errors are reduced to a category derived
  from the exception *type*.
* Emitting telemetry never raises; a failing sink is ignored.

Metric naming (used by the Datadog integration): each event ``<name>`` produces
``vacation_planner.<name>.count`` and one ``vacation_planner.<name>.<measurement>`` per
measurement. Measurements ending in ``_ms`` are latency distributions in milliseconds; the
others are summed counts (for example token usage).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Protocol

METRIC_PREFIX = "vacation_planner"

# Event names
HTTP_REQUEST = "http.request"
AI_GENERATION = "ai.generation"
LLM_REQUEST = "llm.request"
TOOL_EXECUTION = "tool.execution"
MCP_WEATHER_FALLBACK = "mcp.weather.fallback"

# Outcomes and error categories
SUCCESS = "success"
FAILURE = "failure"
NO_ERROR = "none"

ALLOWED_TAG_KEYS = frozenset(
    {
        "method",
        "route",
        "status_code",
        "status_class",
        "outcome",
        "error_category",
        "workflow",
        "model",
        "node",
        "tool",
        "failure_type",
    }
)

_MAX_TAG_VALUE_LENGTH = 100
_INVALID_TAG_CHARS = re.compile(r"[^a-z0-9_.:/\-]")

logger = logging.getLogger("vacation_planner.telemetry")


def normalize_tag_value(value: Any) -> str:
    """Return a short Datadog-compatible tag value; route parameters become ``:name``."""
    text = str(value).strip().casefold().replace("{", ":").replace("}", "")
    text = _INVALID_TAG_CHARS.sub("_", text)[:_MAX_TAG_VALUE_LENGTH]
    return text or "unknown"


@dataclass(frozen=True)
class TelemetryEvent:
    name: str
    tags: Mapping[str, str] = field(default_factory=dict)
    measurements: Mapping[str, float] = field(default_factory=dict)


class TelemetrySink(Protocol):
    def handle(self, event: TelemetryEvent) -> None: ...


class InMemoryTelemetrySink:
    """Collects events; used by tests to assert on emitted telemetry."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []
        self._lock = threading.Lock()

    def handle(self, event: TelemetryEvent) -> None:
        with self._lock:
            self.events.append(event)

    def named(self, name: str) -> list[TelemetryEvent]:
        with self._lock:
            return [event for event in self.events if event.name == name]


class _StderrHandler(logging.StreamHandler):
    """Writes to the *current* ``sys.stderr`` so redirected streams keep working."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:
        pass


def _telemetry_logger() -> logging.Logger:
    if not logger.handlers:
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class LogTelemetrySink:
    """Writes one JSON object per event to the ``vacation_planner.telemetry`` logger."""

    def handle(self, event: TelemetryEvent) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event.name,
            **event.tags,
            **{key: round(value, 3) for key, value in event.measurements.items()},
        }
        _telemetry_logger().info(json.dumps(record, sort_keys=True))


def _configured_sinks() -> list[TelemetrySink]:
    try:
        from app.core.config import settings

        enabled = settings.TELEMETRY_ENABLED
        log_enabled = settings.TELEMETRY_LOG_ENABLED
    except Exception:
        enabled, log_enabled = True, True
    if not enabled:
        return []
    return [LogTelemetrySink()] if log_enabled else []


class Telemetry:
    def __init__(self, sinks: list[TelemetrySink] | None = None) -> None:
        self._sinks = sinks

    @property
    def sinks(self) -> list[TelemetrySink]:
        if self._sinks is None:
            self._sinks = _configured_sinks()
        return self._sinks

    def emit(
        self,
        name: str,
        tags: Mapping[str, Any] | None = None,
        measurements: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            event = TelemetryEvent(
                name=name,
                tags={
                    key: normalize_tag_value(value)
                    for key, value in (tags or {}).items()
                    if key in ALLOWED_TAG_KEYS and value is not None
                },
                measurements={
                    key: float(value)
                    for key, value in (measurements or {}).items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
            )
            sinks = list(self.sinks)
        except Exception:
            return
        for sink in sinks:
            try:
                sink.handle(event)
            except Exception:
                pass

    @contextmanager
    def capture(self) -> Iterator[InMemoryTelemetrySink]:
        """Temporarily replace all sinks with an in-memory sink (for tests)."""
        sink = InMemoryTelemetrySink()
        previous = self._sinks
        self._sinks = [sink]
        try:
            yield sink
        finally:
            self._sinks = previous


telemetry = Telemetry()


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def model_tag(model: Any) -> str:
    """Return the configured model name of a chat model, or ``unknown``."""
    for attribute in ("model", "model_name"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


# ---------------------------------------------------------------------------
# Error categorization (type-based; exception messages are never inspected)
# ---------------------------------------------------------------------------
_EXCEPTION_CATEGORIES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("RateLimit",), "rate_limited"),
    (("Timeout", "TimedOut"), "timeout"),
    (("Authentication", "PermissionDenied"), "auth_error"),
    (("APIConnection", "ConnectError", "ConnectionError"), "connection_error"),
    (("BadRequest",), "bad_request"),
    (("InternalServer", "Overloaded", "ServiceUnavailable", "APIStatusError"), "provider_error"),
    (("MCPClientError",), "mcp_error"),
    (("ValidationError", "OutputParser"), "validation_error"),
)


def _specific_category(exc: BaseException) -> str | None:
    for cls in type(exc).__mro__:
        for markers, category in _EXCEPTION_CATEGORIES:
            if any(marker in cls.__name__ for marker in markers):
                return category
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection_error"
    return None


def exception_category(exc: BaseException) -> str:
    """Categorize an exception by its type, following ``__cause__``/``__context__``.

    A specific category anywhere in the chain (rate limit, timeout, MCP, ...) wins; otherwise
    a ``ValueError`` in the chain is a ``validation_error``; anything else is unexpected.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    for item in chain:
        category = _specific_category(item)
        if category is not None:
            return category
    if any(isinstance(item, ValueError) for item in chain):
        return "validation_error"
    return "unexpected_error"


def http_error_category(status_code: int) -> str:
    if status_code < 400:
        return NO_ERROR
    categories = {
        401: "auth_error",
        403: "auth_error",
        404: "not_found",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "rate_limited",
        502: "upstream_error",
        503: "upstream_error",
        504: "upstream_error",
    }
    if status_code in categories:
        return categories[status_code]
    return "client_error" if status_code < 500 else "server_error"


# ---------------------------------------------------------------------------
# HTTP request middleware
# ---------------------------------------------------------------------------
class RequestTelemetryMiddleware:
    """Pure ASGI middleware recording latency, status and error category per request.

    The route tag is the matched route *template* (for example ``/itineraries/:trip_id``),
    or ``unmatched``; raw paths and query strings are never recorded.
    """

    def __init__(self, app: Any, telemetry_client: Telemetry | None = None) -> None:
        self.app = app
        self.telemetry = telemetry_client or telemetry

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_holder = {"code": None}

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status_holder["code"] = message.get("status")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            self._record(scope, status_holder["code"] or 500, started, "unhandled_exception")
            raise
        self._record(scope, status_holder["code"] or 500, started, None)

    def _record(self, scope, status_code: int, started: float, category: str | None) -> None:
        route = getattr(scope.get("route"), "path", None) or "unmatched"
        category = category or http_error_category(status_code)
        self.telemetry.emit(
            HTTP_REQUEST,
            tags={
                "method": scope.get("method", "unknown"),
                "route": route,
                "status_code": status_code,
                "status_class": f"{status_code // 100}xx",
                "outcome": SUCCESS if category == NO_ERROR else FAILURE,
                "error_category": category,
            },
            measurements={"duration_ms": elapsed_ms(started)},
        )
