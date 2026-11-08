"""Offline tests for the DogStatsD metrics sink and the Datadog dashboard definition.

No Datadog Agent or account is contacted: sockets are fakes, except one loopback test that
sends a datagram to a socket bound by the test itself on 127.0.0.1.
"""

import io
import json
import re
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.core.telemetry import (
    AI_GENERATION,
    EVENT_CATALOG,
    HTTP_REQUEST,
    LLM_REQUEST,
    METRIC_PREFIX,
    DogStatsDSink,
    LogTelemetrySink,
    RequestTelemetryMiddleware,
    Telemetry,
    TelemetryEvent,
    _configured_sinks,
    datadog_metric_lines,
    metric_type,
    telemetry,
)

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = ROOT / "observability" / "datadog-dashboard.json"
SENSITIVE = "secret-user-detail-4242"
GLOBAL_TAG_KEYS = {"service", "env", "version"}


class FakeSocket:
    def __init__(self, error=None):
        self.sent = []
        self.error = error

    def sendto(self, payload, address):
        if self.error:
            raise self.error
        self.sent.append((payload.decode("utf-8"), address))


def fake_sink(**kwargs):
    sock = FakeSocket(kwargs.pop("error", None))
    sink = DogStatsDSink(host="127.0.0.1", port=8125, sock=sock, **kwargs)
    return sink, sock


def catalog_metrics() -> dict[str, tuple[str, str]]:
    """metric name -> (event name, dogstatsd type) for everything the app can send."""
    metrics = {}
    for event, schema in EVENT_CATALOG.items():
        metrics[f"{METRIC_PREFIX}.{event}.count"] = (event, "c")
        for measurement in schema["measurements"]:
            metrics[f"{METRIC_PREFIX}.{event}.{measurement}"] = (event, metric_type(measurement))
    return metrics


class DogStatsDFormattingTests(unittest.TestCase):
    def test_http_event_maps_to_count_and_distribution(self):
        event = TelemetryEvent(
            HTTP_REQUEST,
            tags={"method": "post", "route": "/itineraries", "status_code": "201", "status_class": "2xx",
                  "outcome": "success", "error_category": "none"},
            measurements={"duration_ms": 12.34567},
        )
        tags = "error_category:none,method:post,outcome:success,route:/itineraries,status_class:2xx,status_code:201"
        self.assertEqual(
            datadog_metric_lines(event),
            [
                f"vacation_planner.http.request.count:1|c|#{tags}",
                f"vacation_planner.http.request.duration_ms:12.346|d|#{tags}",
            ],
        )

    def test_token_measurements_are_counts_and_latency_is_a_distribution(self):
        event = TelemetryEvent(
            LLM_REQUEST,
            tags={"model": "claude-haiku-4-5", "node": "plan", "outcome": "success", "error_category": "none"},
            measurements={"duration_ms": 800.0, "input_tokens": 120.0, "output_tokens": 30.0, "total_tokens": 150.0},
        )
        lines = {line.split(":")[0]: line for line in datadog_metric_lines(event)}
        self.assertIn(":800|d|#", lines["vacation_planner.llm.request.duration_ms"])
        self.assertIn(":120|c|#", lines["vacation_planner.llm.request.input_tokens"])
        self.assertIn(":30|c|#", lines["vacation_planner.llm.request.output_tokens"])
        self.assertIn(":150|c|#", lines["vacation_planner.llm.request.total_tokens"])
        self.assertIn(":1|c|#", lines["vacation_planner.llm.request.count"])

    def test_values_are_compact(self):
        for value, text in ((0.0, "0"), (5.0, "5"), (1.5, "1.5"), (0.0004, "0"), (2.0006, "2.001")):
            event = TelemetryEvent(AI_GENERATION, measurements={"duration_ms": value})
            self.assertEqual(datadog_metric_lines(event)[1], f"vacation_planner.ai.generation.duration_ms:{text}|d")

    def test_event_without_tags_or_measurements(self):
        self.assertEqual(
            datadog_metric_lines(TelemetryEvent("mcp.weather.fallback")),
            ["vacation_planner.mcp.weather.fallback.count:1|c"],
        )

    def test_global_tags_are_normalized_and_empty_ones_skipped(self):
        event = TelemetryEvent("mcp.weather.fallback", tags={"failure_type": "timeout"})
        [line] = datadog_metric_lines(
            event, global_tags={"service": "AI Vacation Planner", "env": "Local", "version": ""}
        )
        self.assertEqual(
            line,
            "vacation_planner.mcp.weather.fallback.count:1|c|#env:local,service:ai_vacation_planner,failure_type:timeout",
        )


class DogStatsDPrivacyTests(unittest.TestCase):
    def test_events_tags_and_measurements_outside_the_catalog_are_not_sent(self):
        self.assertEqual(datadog_metric_lines(TelemetryEvent("debug.prompt", tags={"model": "x"})), [])
        event = TelemetryEvent(
            "tool.execution",
            tags={"tool": "travel_weather", "outcome": "success", "error_category": "none", "route": SENSITIVE},
            measurements={"duration_ms": 1.0, "secret_measure": 42.0},
        )
        payload = "\n".join(datadog_metric_lines(event))
        self.assertNotIn(SENSITIVE, payload)
        self.assertNotIn("route:", payload)
        self.assertNotIn("secret_measure", payload)

    def test_http_request_through_middleware_sends_no_raw_path_query_or_message(self):
        sink, sock = fake_sink(global_tags={"service": "ai-vacation-planner"})
        app = FastAPI()
        app.add_middleware(RequestTelemetryMiddleware, telemetry_client=Telemetry([sink]))

        @app.get("/trips/{trip_id}")
        def get_trip(trip_id: int):
            raise RuntimeError(SENSITIVE)

        with TestClient(app, raise_server_exceptions=False) as client:
            client.get(f"/trips/918273?token={SENSITIVE}")

        [(payload, _)] = sock.sent
        self.assertIn("route:/trips/:trip_id", payload)
        self.assertIn("error_category:unhandled_exception", payload)
        self.assertNotIn("918273", payload)
        self.assertNotIn(SENSITIVE, payload)

    def test_orchestrator_metrics_contain_no_prompt_or_tool_input(self):
        from app.services.agent_service import TravelPlanningOrchestrator
        from app.tools import create_weather_tool
        from test_telemetry import ScriptedModel, weather_call

        sink, sock = fake_sink()
        orchestrator = TravelPlanningOrchestrator(
            model=ScriptedModel([weather_call(location=SENSITIVE), AIMessage(content="done")]),
            tools=[create_weather_tool(weather_service=lambda location: {"forecast": "sunny"})],
        )
        with patch.object(telemetry, "_sinks", [sink]):
            orchestrator.invoke(f"Paris trip {SENSITIVE}", expected_destination="Paris", expected_days=1)

        payload = "\n".join(p for p, _ in sock.sent)
        self.assertIn("vacation_planner.tool.execution.count:1|c", payload)
        self.assertIn("vacation_planner.ai.generation.tool_iterations:1|c", payload)
        self.assertNotIn(SENSITIVE, payload)


class DogStatsDSinkTests(unittest.TestCase):
    def test_one_datagram_per_event_with_all_lines(self):
        sink, sock = fake_sink(global_tags={"service": "ai-vacation-planner", "env": "test"})
        sink.handle(TelemetryEvent(AI_GENERATION, tags={"outcome": "success"}, measurements={"duration_ms": 3.0}))
        [(payload, address)] = sock.sent
        self.assertEqual(address[:2], ("127.0.0.1", 8125))
        self.assertEqual(
            payload.split("\n"),
            [
                "vacation_planner.ai.generation.count:1|c|#env:test,service:ai-vacation-planner,outcome:success",
                "vacation_planner.ai.generation.duration_ms:3|d|#env:test,service:ai-vacation-planner,outcome:success",
            ],
        )

    def test_unknown_events_send_nothing(self):
        sink, sock = fake_sink()
        sink.handle(TelemetryEvent("not.in.catalog"))
        self.assertEqual(sock.sent, [])

    def test_send_errors_are_swallowed(self):
        sink, _ = fake_sink(error=OSError("no agent"))
        sink.handle(TelemetryEvent(AI_GENERATION))  # does not raise
        Telemetry([sink]).emit(AI_GENERATION)  # does not raise

    def test_real_udp_datagram_on_loopback(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(5)
        try:
            sink = DogStatsDSink(host="127.0.0.1", port=receiver.getsockname()[1])
            sink.handle(TelemetryEvent("mcp.weather.fallback", tags={"failure_type": "failed"}))
            payload = receiver.recv(65535).decode("utf-8")
        finally:
            receiver.close()
        self.assertEqual(payload, "vacation_planner.mcp.weather.fallback.count:1|c|#failure_type:failed")


class DogStatsDSinkFailureIsolationTests(unittest.TestCase):
    """DNS, socket and formatting failures must never escape the sink."""

    event = TelemetryEvent("mcp.weather.fallback", tags={"failure_type": "failed"})

    def test_dns_resolution_failure_is_swallowed_and_retried_only_after_backoff(self):
        sock = FakeSocket()
        sink = DogStatsDSink(host="agent.invalid", port=8125, sock=sock)
        with patch("app.core.telemetry.socket.getaddrinfo", side_effect=socket.gaierror("no such host")) as resolve:
            sink.handle(self.event)
            sink.handle(self.event)
        self.assertEqual(resolve.call_count, 1, "a failed lookup must not be repeated on every event")
        self.assertEqual(sock.sent, [])

        # Backoff elapsed and the host is now resolvable (numeric address, no DNS needed).
        sink._retry_after = 0.0
        sink.host = "127.0.0.1"
        sink.handle(self.event)
        self.assertEqual(len(sock.sent), 1)
        self.assertEqual(sock.sent[0][1][:2], ("127.0.0.1", 8125))

    def test_socket_creation_failure_is_swallowed(self):
        sink = DogStatsDSink(host="127.0.0.1", port=8125)
        with patch("app.core.telemetry.socket.socket", side_effect=OSError("no sockets")):
            sink.handle(self.event)
        self.assertIsNone(sink._socket)

    def test_unexpected_formatting_failure_is_swallowed(self):
        sink, sock = fake_sink()
        with patch("app.core.telemetry.datadog_metric_lines", side_effect=RuntimeError("bug")):
            sink.handle(self.event)
        self.assertEqual(sock.sent, [])

    def test_failing_dogstatsd_does_not_affect_other_sinks_or_requests(self):
        from app.core.telemetry import InMemoryTelemetrySink

        memory = InMemoryTelemetrySink()
        broken = DogStatsDSink(host="agent.invalid", port=8125)
        client = Telemetry([broken, memory])
        app = FastAPI()
        app.add_middleware(RequestTelemetryMiddleware, telemetry_client=client)

        @app.get("/ok")
        def ok():
            return {"ok": True}

        with patch("app.core.telemetry.socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            with TestClient(app) as test_client:
                response = test_client.get("/ok")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(len(memory.events), 1)


IPV6_LOOPBACK = (socket.AF_INET6, socket.SOCK_DGRAM, 17, "", ("::1", 8125, 0, 0))
IPV4_LOOPBACK = (socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("127.0.0.1", 8125))


class DogStatsDAddressResolutionTests(unittest.TestCase):
    event = TelemetryEvent("mcp.weather.fallback", tags={"failure_type": "failed"})

    def test_prefers_ipv4_when_dns_returns_ipv6_first(self):
        sock = FakeSocket()
        sink = DogStatsDSink(host="localhost", port=8125, sock=sock)
        with patch("app.core.telemetry.socket.getaddrinfo", return_value=[IPV6_LOOPBACK, IPV4_LOOPBACK]):
            sink.handle(self.event)
        [(_, address)] = sock.sent
        self.assertEqual(address, ("127.0.0.1", 8125))

    def test_created_socket_uses_the_ipv4_family(self):
        with (
            patch("app.core.telemetry.socket.getaddrinfo", return_value=[IPV6_LOOPBACK, IPV4_LOOPBACK]),
            patch("app.core.telemetry.socket.socket") as socket_factory,
        ):
            DogStatsDSink(host="localhost", port=8125).handle(self.event)
        socket_factory.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
        created = socket_factory.return_value
        created.setblocking.assert_called_once_with(False)
        [(_, address)] = [call.args for call in created.sendto.call_args_list]
        self.assertEqual(address, ("127.0.0.1", 8125))

    def test_falls_back_to_first_result_when_no_ipv4_is_available(self):
        sock = FakeSocket()
        sink = DogStatsDSink(host="localhost", port=8125, sock=sock)
        with patch("app.core.telemetry.socket.getaddrinfo", return_value=[IPV6_LOOPBACK]):
            sink.handle(self.event)
        [(_, address)] = sock.sent
        self.assertEqual(address, ("::1", 8125, 0, 0))


def parse_tags(line: str) -> list[str]:
    _, _, tag_part = line.partition("|#")
    return tag_part.split(",") if tag_part else []


class DogStatsDTagSafetyTests(unittest.TestCase):
    def test_global_tags_are_restricted_to_service_env_version(self):
        sink, sock = fake_sink(
            global_tags={
                "service": "Planner API",
                "env": "prod",
                "version": "1.2.3",
                "host": "laptop-of-alice",
                "user": "alice@example.com",
                "service,owner": "x",
            }
        )
        self.assertEqual(sink.global_tags, {"service": "planner_api", "env": "prod", "version": "1.2.3"})
        sink.handle(TelemetryEvent("mcp.weather.fallback"))
        [(payload, _)] = sock.sent
        self.assertEqual(parse_tags(payload), ["env:prod", "service:planner_api", "version:1.2.3"])
        self.assertNotIn("alice", payload)

    def test_global_tag_values_cannot_inject_tags(self):
        [line] = datadog_metric_lines(
            TelemetryEvent("mcp.weather.fallback"),
            global_tags={"env": "prod,owner:attacker|#evil", "service": "\nfake.metric:1|c"},
        )
        tags = parse_tags(line)
        self.assertEqual([tag.split(":")[0] for tag in tags], ["env", "service"])
        self.assertEqual(line.count("|"), 2)
        self.assertNotIn("\n", line)

    def test_event_tag_values_cannot_inject_tags_metrics_or_types(self):
        malicious = "x,admin:true|#owner:attacker\nvacation_planner.fake:999|c"
        event = TelemetryEvent(  # built directly, bypassing Telemetry.emit normalization
            "tool.execution",
            tags={"tool": malicious, "outcome": "success", "error_category": "none"},
            measurements={"duration_ms": 1.0},
        )
        lines = datadog_metric_lines(event)
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertNotIn("\n", line)
            self.assertEqual(line.count("|"), 2, line)
            self.assertEqual(line.count("#"), 1, line)
            self.assertEqual(
                sorted(tag.split(":")[0] for tag in parse_tags(line)),
                ["error_category", "outcome", "tool"],
            )
            [tool_tag] = [tag for tag in parse_tags(line) if tag.startswith("tool:")]
            self.assertRegex(tool_tag, r"^tool:[a-z0-9_.:/\-]+$")

    def test_non_finite_and_non_numeric_measurements_are_dropped(self):
        event = TelemetryEvent(
            "llm.request",
            measurements={"duration_ms": float("nan"), "input_tokens": float("inf"), "output_tokens": True, "total_tokens": 5},
        )
        names = [line.split(":")[0] for line in datadog_metric_lines(event)]
        self.assertEqual(names, ["vacation_planner.llm.request.count", "vacation_planner.llm.request.total_tokens"])


class DatadogConfigurationTests(unittest.TestCase):
    def test_datadog_is_disabled_by_default(self):
        defaults = {name: field.default for name, field in Settings.model_fields.items()}
        self.assertIs(defaults["DD_METRICS_ENABLED"], False)
        self.assertEqual(defaults["DD_AGENT_HOST"], "localhost")
        self.assertEqual(defaults["DD_DOGSTATSD_PORT"], 8125)
        self.assertEqual(defaults["DD_SERVICE"], "ai-vacation-planner")
        self.assertEqual(defaults["DD_ENV"], "local")
        self.assertEqual(defaults["DD_VERSION"], "1.0.0")

    def test_env_example_documents_safe_defaults(self):
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        for line in (
            "DD_METRICS_ENABLED=false",
            "DD_AGENT_HOST=localhost",
            "DD_DOGSTATSD_PORT=8125",
            "DD_SERVICE=ai-vacation-planner",
            "DD_ENV=local",
            "DD_VERSION=1.0.0",
            "TELEMETRY_ENABLED=true",
            "TELEMETRY_LOG_ENABLED=true",
        ):
            self.assertIn(line, example)
        self.assertNotRegex(example, r"(?im)^DD_API_KEY\s*=")

    def test_default_sinks_are_json_log_only(self):
        with patch("app.core.config.settings.DD_METRICS_ENABLED", False):
            sinks = _configured_sinks()
        self.assertEqual([type(s) for s in sinks], [LogTelemetrySink])

    def test_enabling_datadog_adds_dogstatsd_and_keeps_json_logs(self):
        with patch.multiple(
            "app.core.config.settings",
            DD_METRICS_ENABLED=True,
            DD_AGENT_HOST="dd-agent",
            DD_DOGSTATSD_PORT=8126,
            DD_SERVICE="planner",
            DD_ENV="staging",
            DD_VERSION="2.0.0",
        ):
            sinks = _configured_sinks()
        log_sink, dd_sink = sinks
        self.assertIsInstance(log_sink, LogTelemetrySink)
        self.assertIsInstance(dd_sink, DogStatsDSink)
        self.assertEqual((dd_sink.host, dd_sink.port), ("dd-agent", 8126))
        self.assertEqual(dd_sink.global_tags, {"service": "planner", "env": "staging", "version": "2.0.0"})

    def test_test_session_is_isolated_from_a_developer_env_enabling_datadog(self):
        import os
        import tempfile

        from app.core.config import settings

        self.assertEqual(os.environ.get("DD_METRICS_ENABLED"), "false")  # set by tests/conftest.py
        self.assertIs(settings.DD_METRICS_ENABLED, False)
        self.assertFalse(any(isinstance(s, DogStatsDSink) for s in Telemetry().sinks))
        self.assertFalse(any(isinstance(s, DogStatsDSink) for s in telemetry.sinks))

        # A .env that enables Datadog is overridden by the test-session environment variable.
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "DATABASE_URL=sqlite://\nSECRET_KEY=test\nDD_METRICS_ENABLED=true\n", encoding="utf-8"
            )
            self.assertIs(Settings(_env_file=env_file).DD_METRICS_ENABLED, False)

    def test_master_switch_disables_datadog_too(self):
        with patch.multiple("app.core.config.settings", TELEMETRY_ENABLED=False, DD_METRICS_ENABLED=True):
            self.assertEqual(_configured_sinks(), [])

    def test_json_log_telemetry_still_written_when_datadog_enabled(self):
        stream = io.StringIO()
        dd_sink, sock = fake_sink()
        with patch.object(sys, "stderr", stream):
            Telemetry([LogTelemetrySink(), dd_sink]).emit(
                AI_GENERATION,
                tags={"workflow": "travel_planning", "outcome": "success"},
                measurements={"duration_ms": 5},
            )
        record = json.loads(stream.getvalue().strip())
        self.assertEqual(record["event"], "ai.generation")
        self.assertEqual(record["workflow"], "travel_planning")
        self.assertEqual(record["duration_ms"], 5.0)
        self.assertEqual(len(sock.sent), 1)


def iter_widgets(widgets):
    for widget in widgets:
        yield widget["definition"]
        yield from iter_widgets(widget["definition"].get("widgets", []))


def dashboard_queries(dashboard):
    for definition in iter_widgets(dashboard["widgets"]):
        for request in definition.get("requests", []):
            for query in request.get("queries", []):
                yield definition, query


QUERY_PATTERN = re.compile(
    r"^(?P<agg>sum|avg|max|min|p50|p75|p90|p95|p99):(?P<metric>[a-z_.]+)"
    r"\{(?P<filters>[^}]*)\}(?: by \{(?P<group>[^}]*)\})?(?P<as_count>\.as_count\(\))?$"
)


class DashboardDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
        cls.metrics = catalog_metrics()

    def test_dashboard_structure(self):
        self.assertTrue(self.dashboard["title"])
        self.assertEqual(self.dashboard["layout_type"], "ordered")
        self.assertEqual(
            {v["name"]: v["prefix"] for v in self.dashboard["template_variables"]},
            {"env": "env", "service": "service"},
        )
        self.assertGreaterEqual(len(list(iter_widgets(self.dashboard["widgets"]))), 20)

    def test_every_query_uses_an_emitted_metric_and_its_tags(self):
        queries = list(dashboard_queries(self.dashboard))
        self.assertGreater(len(queries), 25)
        for definition, query in queries:
            with self.subTest(widget=definition.get("title"), query=query["query"]):
                self.assertEqual(query["data_source"], "metrics")
                match = QUERY_PATTERN.match(query["query"])
                self.assertIsNotNone(match, "unexpected query shape")
                self.assertIn(match["metric"], self.metrics, "metric is not emitted by the app")
                event, kind = self.metrics[match["metric"]]
                allowed_tags = EVENT_CATALOG[event]["tags"] | GLOBAL_TAG_KEYS

                filters = [f.strip() for f in match["filters"].split(",") if f.strip()]
                self.assertIn("$env", filters)
                self.assertIn("$service", filters)
                for item in filters:
                    if not item.startswith("$"):
                        self.assertIn(item.split(":")[0], allowed_tags)
                for tag in (match["group"] or "").split(","):
                    if tag.strip():
                        self.assertIn(tag.strip(), allowed_tags)

                if kind == "c":
                    self.assertEqual(match["agg"], "sum")
                    self.assertTrue(match["as_count"], "count metrics must use .as_count()")
                else:
                    self.assertIn(match["agg"], {"avg", "max", "min", "p50", "p75", "p90", "p95", "p99"})
                    self.assertFalse(match["as_count"])

    def test_formulas_only_reference_defined_queries(self):
        for definition in iter_widgets(self.dashboard["widgets"]):
            for request in definition.get("requests", []):
                names = {q["name"] for q in request["queries"]}
                for formula in request["formulas"]:
                    referenced = set(re.findall(r"[a-z_][a-z0-9_]*", formula["formula"]))
                    self.assertTrue(referenced <= names, (definition["title"], formula))

    def test_every_emitted_event_is_on_the_dashboard(self):
        used = {QUERY_PATTERN.match(q["query"])["metric"] for _, q in dashboard_queries(self.dashboard)}
        used_events = {self.metrics[m][0] for m in used}
        self.assertEqual(used_events, set(EVENT_CATALOG))
        for metric in (
            "vacation_planner.llm.request.input_tokens",
            "vacation_planner.llm.request.output_tokens",
            "vacation_planner.llm.request.duration_ms",
            "vacation_planner.ai.generation.duration_ms",
            "vacation_planner.tool.execution.duration_ms",
            "vacation_planner.http.request.duration_ms",
        ):
            self.assertIn(metric, used)


if __name__ == "__main__":
    unittest.main()
