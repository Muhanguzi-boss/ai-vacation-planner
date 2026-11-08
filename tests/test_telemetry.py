"""Offline tests for application telemetry (no network, LLM, or database access)."""

import io
import json
import logging
import sys
import unittest
from typing import Annotated, TypedDict
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ValidationError

from app.core.telemetry import (
    AI_GENERATION,
    HTTP_REQUEST,
    LLM_REQUEST,
    MCP_WEATHER_FALLBACK,
    TOOL_EXECUTION,
    InMemoryTelemetrySink,
    LogTelemetrySink,
    RequestTelemetryMiddleware,
    Telemetry,
    TelemetryEvent,
    exception_category,
    http_error_category,
    normalize_tag_value,
    telemetry,
)
from app.services.agent_service import TravelPlanningOrchestrator
from app.services.ai_service import GeneratedItinerary
from app.services.llm_telemetry import LLMTelemetryCallback, llm_telemetry_callback
from app.services.mcp_client import MCPClientError
from app.services.vision_service import VisionService
from app.tools import create_travel_knowledge_search_tool, create_weather_tool
from app.tools.mcp_tools import MCPWeatherService

SENSITIVE = "secret-user-detail-4242"
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00fake-jpeg-body"


def all_values(events: list[TelemetryEvent]) -> str:
    return json.dumps(
        [{"name": e.name, **e.tags, **e.measurements} for e in events], default=str
    )


class TelemetryCoreTests(unittest.TestCase):
    def test_tag_values_are_normalized_for_datadog(self):
        self.assertEqual(normalize_tag_value("/itineraries/{trip_id}/audio"), "/itineraries/:trip_id/audio")
        self.assertEqual(normalize_tag_value("POST"), "post")
        self.assertEqual(normalize_tag_value("a b&c"), "a_b_c")
        self.assertEqual(normalize_tag_value(""), "unknown")
        self.assertEqual(len(normalize_tag_value("x" * 500)), 100)

    def test_unknown_tag_keys_and_non_numeric_measurements_are_dropped(self):
        sink = InMemoryTelemetrySink()
        Telemetry([sink]).emit(
            "custom.event",
            tags={"route": "/trips", "prompt": SENSITIVE, "user_email": SENSITIVE, "model": None},
            measurements={"duration_ms": 12, "text": SENSITIVE, "flag": True},
        )
        [event] = sink.events
        self.assertEqual(dict(event.tags), {"route": "/trips"})
        self.assertEqual(dict(event.measurements), {"duration_ms": 12.0})

    def test_failing_sink_never_raises_and_other_sinks_still_receive(self):
        class BrokenSink:
            def handle(self, event):
                raise RuntimeError("sink down")

        sink = InMemoryTelemetrySink()
        Telemetry([BrokenSink(), sink]).emit("custom.event")
        self.assertEqual(len(sink.events), 1)

    def test_capture_replaces_and_restores_sinks(self):
        original = InMemoryTelemetrySink()
        client = Telemetry([original])
        with client.capture() as captured:
            client.emit("inside")
        client.emit("outside")
        self.assertEqual([e.name for e in captured.events], ["inside"])
        self.assertEqual([e.name for e in original.events], ["outside"])

    def test_disabled_settings_produce_no_sinks(self):
        with patch("app.core.config.settings.TELEMETRY_ENABLED", False):
            self.assertEqual(Telemetry().sinks, [])
        with patch("app.core.config.settings.TELEMETRY_LOG_ENABLED", False):
            self.assertEqual(Telemetry().sinks, [])
        self.assertIsInstance(Telemetry().sinks[0], LogTelemetrySink)

    def test_http_error_categories(self):
        expected = {
            200: "none",
            201: "none",
            400: "client_error",
            401: "auth_error",
            403: "auth_error",
            404: "not_found",
            413: "payload_too_large",
            415: "unsupported_media_type",
            422: "validation_error",
            429: "rate_limited",
            500: "server_error",
            502: "upstream_error",
            504: "upstream_error",
        }
        for status_code, category in expected.items():
            self.assertEqual(http_error_category(status_code), category, status_code)

    def test_exception_categories_use_types_and_cause_chain(self):
        class RateLimitError(Exception):
            pass

        class APITimeoutError(Exception):
            pass

        class AuthenticationError(Exception):
            pass

        class Model(BaseModel):
            value: int

        try:
            Model(value="not-a-number")
        except ValidationError as exc:
            pydantic_error = exc

        try:
            try:
                raise MCPClientError(f"timed out for {SENSITIVE}")
            except MCPClientError as inner:
                raise RuntimeError("Weather lookup failed") from inner
        except RuntimeError as exc:
            wrapped_mcp = exc

        self.assertEqual(exception_category(RateLimitError()), "rate_limited")
        self.assertEqual(exception_category(APITimeoutError()), "timeout")
        self.assertEqual(exception_category(AuthenticationError()), "auth_error")
        self.assertEqual(exception_category(TimeoutError()), "timeout")
        self.assertEqual(exception_category(pydantic_error), "validation_error")
        self.assertEqual(exception_category(ValueError("bad")), "validation_error")
        self.assertEqual(exception_category(wrapped_mcp), "mcp_error")
        self.assertEqual(exception_category(KeyError("x")), "unexpected_error")


class LogSinkTests(unittest.TestCase):
    def test_log_sink_writes_one_json_object_per_event(self):
        stream = io.StringIO()
        with patch.object(sys, "stderr", stream):
            Telemetry([LogTelemetrySink()]).emit(
                HTTP_REQUEST,
                tags={"route": "/trips/{trip_id}", "outcome": "success"},
                measurements={"duration_ms": 1.23456},
            )
        record = json.loads(stream.getvalue().strip())
        self.assertEqual(record["event"], "http.request")
        self.assertEqual(record["route"], "/trips/:trip_id")
        self.assertEqual(record["duration_ms"], 1.235)
        self.assertIn("timestamp", record)


class RequestTelemetryMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.sink = InMemoryTelemetrySink()
        app = FastAPI()
        app.add_middleware(RequestTelemetryMiddleware, telemetry_client=Telemetry([self.sink]))

        @app.get("/items/{item_id}")
        def get_item(item_id: int):
            return {"item_id": item_id}

        @app.post("/fail")
        def fail():
            raise HTTPException(status_code=502, detail="AI generation failed")

        @app.get("/crash")
        def crash():
            raise RuntimeError(SENSITIVE)

        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        self.client.close()

    def only_event(self):
        [event] = self.sink.named(HTTP_REQUEST)
        return event

    def test_success_uses_route_template_not_raw_path_or_query(self):
        response = self.client.get(f"/items/987654?q={SENSITIVE}")
        self.assertEqual(response.status_code, 200)
        event = self.only_event()
        self.assertEqual(
            {k: event.tags[k] for k in ("method", "route", "status_code", "status_class", "outcome", "error_category")},
            {
                "method": "get",
                "route": "/items/:item_id",
                "status_code": "200",
                "status_class": "2xx",
                "outcome": "success",
                "error_category": "none",
            },
        )
        self.assertGreaterEqual(event.measurements["duration_ms"], 0)
        self.assertNotIn("987654", all_values(self.sink.events))
        self.assertNotIn(SENSITIVE, all_values(self.sink.events))

    def test_handled_upstream_failure(self):
        self.client.post("/fail")
        event = self.only_event()
        self.assertEqual(event.tags["status_code"], "502")
        self.assertEqual(event.tags["outcome"], "failure")
        self.assertEqual(event.tags["error_category"], "upstream_error")

    def test_request_validation_failure(self):
        self.client.get("/items/not-an-int")
        self.assertEqual(self.only_event().tags["error_category"], "validation_error")

    def test_unmatched_route(self):
        self.client.get(f"/{SENSITIVE}")
        event = self.only_event()
        self.assertEqual(event.tags["route"], "unmatched")
        self.assertEqual(event.tags["error_category"], "not_found")
        self.assertNotIn(SENSITIVE, all_values(self.sink.events))

    def test_unhandled_exception_is_recorded_without_its_message(self):
        response = self.client.get("/crash")
        self.assertEqual(response.status_code, 500)
        event = self.only_event()
        self.assertEqual(event.tags["status_code"], "500")
        self.assertEqual(event.tags["error_category"], "unhandled_exception")
        self.assertNotIn(SENSITIVE, all_values(self.sink.events))

    def test_application_registers_the_middleware(self):
        from pathlib import Path

        source = Path(__file__).resolve().parents[1].joinpath("app", "main.py").read_text(encoding="utf-8")
        self.assertIn("app.add_middleware(RequestTelemetryMiddleware)", source)


class LLMTelemetryCallbackTests(unittest.TestCase):
    def setUp(self):
        self.sink = InMemoryTelemetrySink()
        self.callback = LLMTelemetryCallback(Telemetry([self.sink]))

    def fake_model(self, messages):
        return GenericFakeChatModel(messages=iter(messages), callbacks=[self.callback])

    def test_records_model_tokens_and_latency_without_prompt_or_completion(self):
        model = self.fake_model(
            [
                AIMessage(
                    content=f"completion {SENSITIVE}",
                    usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
                    response_metadata={"model_name": "claude-haiku-4-5-20251001"},
                )
            ]
        )
        model.invoke([HumanMessage(content=f"prompt {SENSITIVE}")])

        [event] = self.sink.named(LLM_REQUEST)
        self.assertEqual(event.tags["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(event.tags["node"], "standalone")
        self.assertEqual(event.tags["outcome"], "success")
        self.assertEqual(event.measurements["input_tokens"], 120)
        self.assertEqual(event.measurements["output_tokens"], 30)
        self.assertEqual(event.measurements["total_tokens"], 150)
        self.assertGreaterEqual(event.measurements["duration_ms"], 0)
        self.assertNotIn(SENSITIVE, all_values(self.sink.events))

    def test_missing_usage_is_omitted_not_invented(self):
        self.fake_model([AIMessage(content="ok")]).invoke("hello")
        [event] = self.sink.named(LLM_REQUEST)
        self.assertNotIn("input_tokens", event.measurements)
        self.assertIn("duration_ms", event.measurements)

    def test_records_failure_category(self):
        class RateLimitError(Exception):
            pass

        def explode():
            raise RateLimitError(SENSITIVE)
            yield  # pragma: no cover

        with self.assertRaises(RateLimitError):
            GenericFakeChatModel(messages=explode(), callbacks=[self.callback]).invoke("hi")

        [event] = self.sink.named(LLM_REQUEST)
        self.assertEqual(event.tags["outcome"], "failure")
        self.assertEqual(event.tags["error_category"], "rate_limited")
        self.assertNotIn(SENSITIVE, all_values(self.sink.events))

    def test_langgraph_node_name_is_tagged(self):
        model = self.fake_model([AIMessage(content="ok")])

        class State(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]

        builder = StateGraph(State)
        builder.add_node("plan", lambda state: {"messages": [model.invoke(state["messages"])]})
        builder.add_edge(START, "plan")
        builder.add_edge("plan", END)
        builder.compile().invoke({"messages": [HumanMessage(content="hi")]})

        [event] = self.sink.named(LLM_REQUEST)
        self.assertEqual(event.tags["node"], "plan")

    def test_production_anthropic_model_carries_the_callback(self):
        with patch("app.services.agent_service.ai_service.anthropic_key", "test-key-not-real"):
            model = TravelPlanningOrchestrator._create_anthropic_model()
        self.assertIn(llm_telemetry_callback, model.callbacks)


def itinerary(destination="Paris", day_count=1) -> GeneratedItinerary:
    return GeneratedItinerary.model_validate(
        {
            "destination": destination,
            "days": [
                {
                    "day": day,
                    "activities": [
                        {"time": "morning", "activity": "Museum", "location": destination, "estimated_cost": "$10"}
                    ],
                }
                for day in range(1, day_count + 1)
            ],
            "total_estimated_cost": "$10",
        }
    )


class ScriptedModel:
    """Fake chat model: scripted planner responses, then a fixed final itinerary."""

    model = "claude-test-model"

    def __init__(self, planner_responses, final=None):
        self.planner_responses = iter(planner_responses)
        self.final = final or itinerary()

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        outer = self

        class Finalizer:
            def invoke(self, messages):
                return outer.final

        return Finalizer()

    def invoke(self, messages):
        return next(self.planner_responses)


def weather_call(call_id="call-1", location=SENSITIVE):
    return AIMessage(
        content="",
        tool_calls=[{"name": "travel_weather", "args": {"location": location}, "id": call_id, "type": "tool_call"}],
    )


class OrchestratorTelemetryTests(unittest.TestCase):
    def orchestrator(self, model, weather=lambda location: {"forecast": "sunny"}):
        tools = [
            create_travel_knowledge_search_tool(knowledge_service=type("K", (), {"search": lambda self, q, top_k=5: []})()),
            create_weather_tool(weather_service=weather),
        ]
        return TravelPlanningOrchestrator(model=model, tools=tools)

    def test_successful_generation_with_tool_call(self):
        model = ScriptedModel([weather_call(), AIMessage(content="done")])
        with telemetry.capture() as sink:
            self.orchestrator(model).invoke(f"Trip to Paris {SENSITIVE}", expected_destination="Paris", expected_days=1)

        [tool_event] = sink.named(TOOL_EXECUTION)
        self.assertEqual(tool_event.tags["tool"], "travel_weather")
        self.assertEqual(tool_event.tags["outcome"], "success")
        self.assertGreaterEqual(tool_event.measurements["duration_ms"], 0)

        [generation] = sink.named(AI_GENERATION)
        self.assertEqual(generation.tags["workflow"], "travel_planning")
        self.assertEqual(generation.tags["model"], "claude-test-model")
        self.assertEqual(generation.tags["outcome"], "success")
        self.assertEqual(generation.tags["error_category"], "none")
        self.assertEqual(generation.measurements["tool_iterations"], 1)
        self.assertNotIn(SENSITIVE, all_values(sink.events))

    def test_tool_failure_is_recorded_as_tool_error(self):
        def broken_weather(location):
            raise TimeoutError(f"upstream slow for {location}")

        model = ScriptedModel([weather_call()])
        with telemetry.capture() as sink:
            with self.assertRaises(Exception):
                self.orchestrator(model, weather=broken_weather).invoke("Trip to Paris")

        [tool_event] = sink.named(TOOL_EXECUTION)
        self.assertEqual(tool_event.tags["outcome"], "failure")
        self.assertEqual(tool_event.tags["error_category"], "timeout")
        [generation] = sink.named(AI_GENERATION)
        self.assertEqual(generation.tags["outcome"], "failure")
        self.assertEqual(generation.tags["error_category"], "tool_error")
        self.assertNotIn(SENSITIVE, all_values(sink.events))

    def test_tool_limit_is_categorized(self):
        model = ScriptedModel([weather_call(f"call-{i}") for i in range(10)])
        with telemetry.capture() as sink:
            with self.assertRaises(Exception):
                self.orchestrator(model).invoke("Trip to Paris")
        [generation] = sink.named(AI_GENERATION)
        self.assertEqual(generation.tags["error_category"], "tool_limit")

    def test_constraint_violation_is_a_validation_error(self):
        model = ScriptedModel([AIMessage(content="done")], final=itinerary("Rome"))
        with telemetry.capture() as sink:
            with self.assertRaises(Exception):
                self.orchestrator(model).invoke("Trip to Paris", expected_destination="Paris")
        [generation] = sink.named(AI_GENERATION)
        self.assertEqual(generation.tags["outcome"], "failure")
        self.assertEqual(generation.tags["error_category"], "validation_error")


class FakeVisionModel:
    model = "claude-test-model"

    def __init__(self, error=None):
        self.error = error

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        if self.error:
            raise self.error
        return {
            "is_travel_related": True,
            "likely_destination": "Paris, France",
            "confidence": 0.9,
            "landmarks": [],
            "setting": "urban",
            "suggested_trip_style": "culture",
            "description": f"A photo {SENSITIVE}",
        }


class VisionAndMcpTelemetryTests(unittest.TestCase):
    def test_vision_success_and_failure(self):
        with telemetry.capture() as sink:
            VisionService(model=FakeVisionModel()).analyze(JPEG_BYTES, "image/jpeg")
            with self.assertRaises(Exception), self.assertLogs("app.services.vision_service", "ERROR") as logs:
                VisionService(model=FakeVisionModel(error=ValueError(SENSITIVE))).analyze(JPEG_BYTES, "image/jpeg")

        success, failure = sink.named(AI_GENERATION)
        self.assertEqual(success.tags["workflow"], "vision_analysis")
        self.assertEqual(success.tags["model"], "claude-test-model")
        self.assertEqual(success.tags["outcome"], "success")
        self.assertEqual(failure.tags["outcome"], "failure")
        self.assertEqual(failure.tags["error_category"], "validation_error")
        self.assertNotIn(SENSITIVE, all_values(sink.events))
        self.assertNotIn(SENSITIVE, "\n".join(logs.output))

    def test_mcp_fallback_is_counted(self):
        class FailingClient:
            def call_tool(self, name, arguments):
                raise MCPClientError("MCP tool 'get_forecast' timed out")

        service = MCPWeatherService(client=FailingClient(), fallback=lambda location: {"forecast": "mild"})
        with telemetry.capture() as sink, self.assertLogs("app.tools.mcp_tools", "WARNING"):
            service.get_weather(SENSITIVE)
        [event] = sink.named(MCP_WEATHER_FALLBACK)
        self.assertEqual(dict(event.tags), {"failure_type": "timeout"})


class ControllerLoggingTests(unittest.TestCase):
    def test_database_error_log_omits_exception_message(self):
        from unittest.mock import MagicMock

        from fastapi import HTTPException as FastAPIHTTPException

        from app.controllers import itinerary_controller
        from app.models import itinerary as _itinerary, trip as _trip, user as _user  # noqa: F401  (configure mappers)

        db = MagicMock()
        trip = MagicMock(id=1, destination="Paris", days=1, trip_style=None, budget=None)
        db.query.return_value.filter.return_value.first.side_effect = [trip, None]
        db.commit.side_effect = RuntimeError(f"INSERT ... parameters {SENSITIVE}")

        with patch.object(itinerary_controller, "TravelPlanningOrchestrator") as orchestrator:
            orchestrator.return_value.invoke.return_value = itinerary()
            with self.assertLogs("app.controllers.itinerary_controller", logging.CRITICAL) as logs:
                with self.assertRaises(FastAPIHTTPException):
                    itinerary_controller.generate_and_save_itinerary(db, 1, 1)

        output = "\n".join(logs.output)
        self.assertIn("RuntimeError", output)
        self.assertNotIn(SENSITIVE, output)


if __name__ == "__main__":
    unittest.main()
