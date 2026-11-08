"""Offline tests for .env loading and AI-lifecycle failure telemetry.

No live Anthropic call is made: the API key is absent or fake, and chat models are fakes.
"""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.core.env import DEFAULT_ENV_FILE, PROJECT_ROOT, load_local_env
from app.core.telemetry import AI_GENERATION, LLM_MODEL_INIT, LLM_REQUEST, TOOL_EXECUTION, telemetry
from app.services.agent_service import (
    ModelInitializationError,
    OrchestrationError,
    TravelPlanningOrchestrator,
)
from app.services.ai_service import AIService
from app.services.llm_telemetry import llm_telemetry_callback
from app.services.vision_service import VisionAnalysisError, VisionService
from app.tools import create_weather_tool

SECRET = "sk-ant-test-secret-0000"
SENSITIVE = "secret-user-detail-4242"
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00fake-jpeg-body"
ROOT = Path(__file__).resolve().parents[1]
ANTHROPIC_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS")


def dumped(events) -> str:
    return json.dumps([{"name": e.name, **e.tags, **e.measurements} for e in events])


def without_anthropic_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in ANTHROPIC_VARS and k.lower() not in
            {name.lower() for name in ANTHROPIC_VARS}}


class EnvLoadingTests(unittest.TestCase):
    def write_env(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_default_env_file_is_the_project_root_env(self):
        self.assertEqual(DEFAULT_ENV_FILE, PROJECT_ROOT / ".env")
        self.assertTrue((PROJECT_ROOT / "app" / "main.py").is_file())

    def test_anthropic_configuration_is_loaded_into_the_environment(self):
        path = self.write_env(
            f"ANTHROPIC_API_KEY={SECRET}\nANTHROPIC_MODEL=claude-test-model\n"
            "LLM_TEMPERATURE=0.1\nLLM_MAX_TOKENS=1234\n"
        )
        with patch.dict(os.environ, without_anthropic_env(), clear=True):
            self.assertTrue(load_local_env(path))
            self.assertEqual(os.environ["ANTHROPIC_API_KEY"], SECRET)
            service = AIService()
        self.assertEqual(service.anthropic_key, SECRET)
        self.assertEqual(service.anthropic_model, "claude-test-model")
        self.assertEqual(service.temperature, 0.1)
        self.assertEqual(service.max_tokens, 1234)

    def test_existing_environment_variables_take_precedence(self):
        path = self.write_env(f"ANTHROPIC_API_KEY={SECRET}\nANTHROPIC_MODEL=from-file\n")
        env = {**without_anthropic_env(), "ANTHROPIC_MODEL": "from-shell"}
        with patch.dict(os.environ, env, clear=True):
            load_local_env(path)
            self.assertEqual(os.environ["ANTHROPIC_MODEL"], "from-shell")
            self.assertEqual(os.environ["ANTHROPIC_API_KEY"], SECRET)

    def test_missing_file_is_a_no_op(self):
        with patch.dict(os.environ, without_anthropic_env(), clear=True):
            self.assertFalse(load_local_env(ROOT / "does-not-exist.env"))
            self.assertNotIn("ANTHROPIC_API_KEY", os.environ)

    def test_main_loads_env_before_importing_application_modules(self):
        tree = ast.parse((ROOT / "app" / "main.py").read_text(encoding="utf-8"))
        body = tree.body
        load_call = next(
            index for index, node in enumerate(body)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "load_local_env"
        )
        first_app_import = next(
            index for index, node in enumerate(body)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app.")
            and node.module != "app.core.env"
        )
        self.assertLess(load_call, first_app_import)

    def test_deepeval_suite_loads_env_and_keeps_anthropic(self):
        source = (ROOT / "tests" / "evaluation" / "deepeval_itinerary_eval.py").read_text(encoding="utf-8")
        self.assertIn("load_local_env()", source)
        self.assertIn("from langchain_anthropic import ChatAnthropic", source)
        self.assertNotIn("openai", source.casefold())


class ModelInitializationTelemetryTests(unittest.TestCase):
    def test_missing_api_key(self):
        with patch("app.services.agent_service.ai_service.anthropic_key", None), telemetry.capture() as sink:
            with self.assertRaises(ModelInitializationError) as raised:
                TravelPlanningOrchestrator(tools=[create_weather_tool(lambda location: {})])

        self.assertIsInstance(raised.exception, ValueError)  # backwards compatible
        [init] = sink.named(LLM_MODEL_INIT)
        self.assertEqual(init.tags["outcome"], "failure")
        self.assertEqual(init.tags["error_category"], "missing_api_key")
        [generation] = sink.named(AI_GENERATION)
        self.assertEqual(generation.tags["workflow"], "travel_planning")
        self.assertEqual(generation.tags["outcome"], "failure")
        self.assertEqual(generation.tags["error_category"], "model_init_error")

    def test_constructor_failure_hides_provider_details(self):
        with (
            patch("app.services.agent_service.ai_service.anthropic_key", SECRET),
            patch("app.services.agent_service.ChatAnthropic", side_effect=ValueError(f"bad key {SECRET}")),
            telemetry.capture() as sink,
        ):
            with self.assertRaises(ModelInitializationError) as raised:
                TravelPlanningOrchestrator._create_anthropic_model()

        self.assertNotIn(SECRET, str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)
        [init] = sink.named(LLM_MODEL_INIT)
        self.assertEqual(init.tags["error_category"], "validation_error")
        self.assertNotIn(SECRET, dumped(sink.events))

    def test_successful_initialization_is_counted(self):
        with patch("app.services.agent_service.ai_service.anthropic_key", SECRET), telemetry.capture() as sink:
            TravelPlanningOrchestrator._create_anthropic_model()
        [init] = sink.named(LLM_MODEL_INIT)
        self.assertEqual(init.tags["outcome"], "success")
        self.assertEqual(init.tags["error_category"], "none")
        self.assertNotIn(SECRET, dumped(sink.events))

    def test_vision_model_initialization_failure(self):
        with patch("app.services.agent_service.ai_service.anthropic_key", None), telemetry.capture() as sink:
            with self.assertRaises(VisionAnalysisError):
                VisionService().analyze(JPEG_BYTES, "image/jpeg")
        [generation] = sink.named(AI_GENERATION)
        self.assertEqual(generation.tags["workflow"], "vision_analysis")
        self.assertEqual(generation.tags["error_category"], "model_init_error")
        self.assertEqual(sink.named(LLM_MODEL_INIT)[0].tags["error_category"], "missing_api_key")

    def test_itinerary_endpoint_path_records_failure_and_returns_generic_502(self):
        from app.controllers import itinerary_controller
        from app.models import itinerary as _itinerary, trip as _trip, user as _user  # noqa: F401

        db = MagicMock()
        trip = MagicMock(id=1, destination="Paris", days=1, trip_style=None, budget=None)
        db.query.return_value.filter.return_value.first.side_effect = [trip, None]

        with (
            patch("app.services.agent_service.ai_service.anthropic_key", None),
            patch("app.services.agent_service.default_travel_tools", return_value=[]),
            telemetry.capture() as sink,
            self.assertLogs("app.controllers.itinerary_controller", "ERROR") as logs,
        ):
            with self.assertRaises(HTTPException) as raised:
                itinerary_controller.generate_and_save_itinerary(db, 1, 1)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail, "AI generation failed")
        self.assertNotIn("ANTHROPIC_API_KEY", "\n".join(logs.output))
        self.assertEqual(sink.named(AI_GENERATION)[0].tags["error_category"], "model_init_error")


class RateLimitError(Exception):
    """Named like the Anthropic SDK error; categorized by type name."""


class FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RateLimitError(f"provider said no for {SENSITIVE}")

    def bind_tools(self, tools, **kwargs):
        return self


class GenerationFailureTelemetryTests(unittest.TestCase):
    def test_llm_failure_is_recorded_per_call_and_per_generation(self):
        model = FailingChatModel(callbacks=[llm_telemetry_callback])
        orchestrator = TravelPlanningOrchestrator(
            model=model, tools=[create_weather_tool(lambda location: {})]
        )
        with telemetry.capture() as sink:
            with self.assertRaises(OrchestrationError):
                orchestrator.invoke(f"Plan Paris {SENSITIVE}")

        [llm] = sink.named(LLM_REQUEST)
        self.assertEqual(llm.tags["node"], "plan")
        self.assertEqual(llm.tags["outcome"], "failure")
        self.assertEqual(llm.tags["error_category"], "rate_limited")
        self.assertIn("duration_ms", llm.measurements)
        [generation] = sink.named(AI_GENERATION)
        self.assertEqual(generation.tags["error_category"], "rate_limited")
        self.assertNotIn(SENSITIVE, dumped(sink.events))

    def test_unsupported_tool_request_is_recorded_without_its_name(self):
        from test_telemetry import ScriptedModel

        requested = f"delete_everything_{SENSITIVE}"
        model = ScriptedModel(
            [
                AIMessage(content="", tool_calls=[{"name": requested, "args": {}, "id": "c1", "type": "tool_call"}]),
                AIMessage(content="done"),
            ]
        )
        orchestrator = TravelPlanningOrchestrator(model=model, tools=[create_weather_tool(lambda location: {})])
        with telemetry.capture() as sink:
            orchestrator.invoke("Plan Paris", expected_destination="Paris", expected_days=1)

        [tool_event] = sink.named(TOOL_EXECUTION)
        self.assertEqual(
            dict(tool_event.tags),
            {"tool": "unsupported", "outcome": "failure", "error_category": "unsupported_tool"},
        )
        self.assertNotIn(SENSITIVE, dumped(sink.events))
        self.assertNotIn("delete_everything", dumped(sink.events))
        self.assertEqual(sink.named(AI_GENERATION)[0].tags["outcome"], "success")


class CatalogConformanceTests(unittest.TestCase):
    def test_events_from_real_code_paths_match_the_catalog(self):
        from app.core.telemetry import EVENT_CATALOG
        from test_telemetry import FakeVisionModel, ScriptedModel, weather_call
        from app.services.mcp_client import MCPClientError
        from app.tools.mcp_tools import MCPWeatherService

        class FailingClient:
            def call_tool(self, name, arguments):
                raise MCPClientError("MCP tool 'get_forecast' failed")

        with telemetry.capture() as sink, self.assertLogs(level="WARNING"):
            TravelPlanningOrchestrator(
                model=ScriptedModel([weather_call(location="Paris"), AIMessage(content="done")]),
                tools=[create_weather_tool(lambda location: {"forecast": "sunny"})],
            ).invoke("Plan Paris", expected_destination="Paris", expected_days=1)
            VisionService(model=FakeVisionModel()).analyze(JPEG_BYTES, "image/jpeg")
            MCPWeatherService(client=FailingClient(), fallback=lambda location: {}).get_weather("Paris")
            with patch("app.services.agent_service.ai_service.anthropic_key", SECRET):
                TravelPlanningOrchestrator._create_anthropic_model()
            with self.assertRaises(OrchestrationError):
                TravelPlanningOrchestrator(
                    model=FailingChatModel(callbacks=[llm_telemetry_callback]),
                    tools=[create_weather_tool(lambda location: {})],
                ).invoke("Plan Paris")

        names = {event.name for event in sink.events}
        self.assertEqual(names, set(EVENT_CATALOG) - {"http.request"})
        for event in sink.events:
            with self.subTest(event=event.name):
                schema = EVENT_CATALOG[event.name]
                self.assertTrue(set(event.tags) <= schema["tags"], event.tags)
                self.assertTrue(set(event.measurements) <= schema["measurements"], event.measurements)


if __name__ == "__main__":
    unittest.main()
