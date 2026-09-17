import unittest

from langchain_core.messages import AIMessage

from app.services.ai_service import GeneratedItinerary
from app.services.agent_service import (
    MAX_TOOL_ITERATIONS,
    OrchestrationError,
    TravelPlanningOrchestrator,
)
from app.tools import create_travel_knowledge_search_tool, create_weather_tool


def sample_itinerary() -> GeneratedItinerary:
    return GeneratedItinerary.model_validate(
        {
            "destination": "Paris",
            "days": [
                {
                    "day": 1,
                    "activities": [
                        {
                            "time": "morning",
                            "activity": "Visit the Louvre",
                            "location": "Paris",
                            "estimated_cost": "$25",
                        }
                    ],
                }
            ],
            "total_estimated_cost": "$25",
        }
    )


class FakeModel:
    def __init__(self, planner_responses, final_response=None):
        self.planner_responses = iter(planner_responses)
        self.final_response = final_response or sample_itinerary()
        self.planner_inputs = []
        self.finalization_inputs = []

    def bind_tools(self, tools):
        self.tools = tools
        return FakePlanner(self)

    def with_structured_output(self, schema):
        self.output_schema = schema
        return FakeFinalizer(self)

    def invoke(self, messages):
        self.planner_inputs.append(messages)
        if self.planner_responses:
            try:
                return next(self.planner_responses)
            except StopIteration:
                pass
        self.finalization_inputs.append(messages)
        if callable(self.final_response):
            return self.final_response(messages)
        return self.final_response


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
        if callable(self.model.final_response):
            return self.model.final_response(messages)
        return self.model.final_response


class AgentServiceTests(unittest.TestCase):
    def test_graph_constructs_and_returns_structured_output_without_tools(self):
        model = FakeModel([AIMessage(content="No tools needed")])

        result = TravelPlanningOrchestrator(model=model).invoke(
            "Create a simple 3-day Paris itinerary."
        )

        self.assertIsInstance(result, GeneratedItinerary)
        self.assertEqual(result.destination, "Paris")
        self.assertEqual(model.output_schema, GeneratedItinerary)

    def test_graph_invokes_knowledge_tool_and_passes_result_back(self):
        model = FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "travel_knowledge_search",
                            "args": {"query": "Paris museums", "top_k": 1},
                            "id": "knowledge-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Use the retrieved travel context."),
            ]
        )
        knowledge_tool = create_travel_knowledge_search_tool(
            KnowledgeServiceStub()
        )

        result = TravelPlanningOrchestrator(model=model, tools=[knowledge_tool]).invoke(
            "Plan a museum-focused Paris trip."
        )

        self.assertEqual(result.destination, "Paris")
        self.assertTrue(
            any(
                "Paris museums are concentrated near the Seine." in str(messages)
                for messages in model.planner_inputs[1:]
            )
        )

    def test_graph_invokes_weather_tool(self):
        model = FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "travel_weather",
                            "args": {"location": "Paris"},
                            "id": "weather-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Use the forecast."),
            ]
        )
        weather_tool = create_weather_tool(
            lambda location: {"summary": "Paris weather forecast: sunny, 24°C."}
        )

        result = TravelPlanningOrchestrator(model=model, tools=[weather_tool]).invoke(
            "Plan Paris activities around the weather."
        )

        self.assertEqual(result.destination, "Paris")
        self.assertTrue(
            any(
                "Paris weather forecast: sunny, 24°C." in str(messages)
                for messages in model.planner_inputs[1:]
            )
        )

    def test_finalization_receives_request_and_tool_result(self):
        model = FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "travel_weather",
                            "args": {"location": "Paris"},
                            "id": "weather-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The forecast is useful for planning."),
            ],
            final_response=lambda messages: (
                sample_itinerary()
                if "Plan around this exact forecast" in str(messages)
                and "Paris weather forecast: sunny, 24°C." in str(messages)
                else (_ for _ in ()).throw(AssertionError("finalization lost workflow context"))
            ),
        )
        weather_tool = create_weather_tool(
            lambda location: {"forecast": "Paris weather forecast: sunny, 24°C."}
        )

        result = TravelPlanningOrchestrator(model=model, tools=[weather_tool]).invoke(
            "Plan around this exact forecast"
        )

        self.assertEqual(result.destination, "Paris")
        self.assertEqual(len(model.finalization_inputs), 1)

    def test_tool_failures_are_wrapped(self):
        model = FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "travel_weather",
                            "args": {"location": "Paris"},
                            "id": "weather-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        failing_tool = create_weather_tool(
            lambda location: (_ for _ in ()).throw(OSError("unavailable"))
        )

        with self.assertRaises(OrchestrationError):
            TravelPlanningOrchestrator(model=model, tools=[failing_tool]).invoke(
                "Plan Paris activities around the weather."
            )

    def test_tool_loop_stops_at_configured_limit(self):
        def repeated_tool_calls():
            iteration = 0
            while True:
                iteration += 1
                yield AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "travel_weather",
                            "args": {"location": "Paris"},
                            "id": f"repeated-weather-call-{iteration}",
                            "type": "tool_call",
                        }
                    ],
                )

        model = FakeModel(repeated_tool_calls())
        weather_tool = create_weather_tool(lambda location: {"summary": "sunny"})

        with self.assertRaises(OrchestrationError) as context:
            TravelPlanningOrchestrator(model=model, tools=[weather_tool]).invoke(
                "Keep checking the weather for Paris."
            )

        self.assertIn("tool limit", str(context.exception))


class KnowledgeServiceStub:
    def search(self, query, top_k=5):
        return [
            {
                "text": "Paris museums are concentrated near the Seine.",
                "metadata": {"source": "stub"},
                "score": 0.9,
            }
        ]


if __name__ == "__main__":
    unittest.main()