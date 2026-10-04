import unittest
from unittest.mock import Mock

from app.tools.travel_tools import (
    create_travel_knowledge_search_tool,
    create_weather_tool,
)


class TravelToolTests(unittest.TestCase):
    def test_knowledge_tool_delegates_and_preserves_results(self):
        service = Mock()
        service.search.return_value = [
            {
                "text": "Use the metro in central Paris.",
                "metadata": {"source": "guide"},
                "score": 0.91,
            }
        ]
        tool = create_travel_knowledge_search_tool(service)

        result = tool.invoke({"query": "Paris transportation", "top_k": 3})

        service.search.assert_called_once_with("Paris transportation", top_k=3)
        self.assertEqual(result["query"], "Paris transportation")
        self.assertEqual(result["results"][0]["metadata"]["source"], "guide")

    def test_weather_tool_delegates_location(self):
        service = Mock()
        service.get_weather.return_value = {"forecast": "sunny", "temperature": 24}
        tool = create_weather_tool(service)

        result = tool.invoke({"location": "Lisbon"})

        service.get_weather.assert_called_once_with("Lisbon")
        self.assertEqual(result["location"], "Lisbon")
        self.assertEqual(result["weather"]["forecast"], "sunny")

    def test_tools_validate_required_inputs(self):
        knowledge_tool = create_travel_knowledge_search_tool(Mock())
        weather_tool = create_weather_tool(Mock())

        with self.assertRaises(ValueError):
            knowledge_tool.invoke({"query": "   "})
        with self.assertRaises(ValueError):
            weather_tool.invoke({"location": ""})

    def test_underlying_failures_are_reported_as_tool_errors(self):
        service = Mock()
        service.search.side_effect = OSError("store unavailable")
        tool = create_travel_knowledge_search_tool(service)

        with self.assertRaises(RuntimeError) as context:
            tool.invoke({"query": "Rome history"})

        self.assertIn("Travel knowledge search failed", str(context.exception))

    def test_weather_failures_are_reported_as_tool_errors(self):
        service = Mock()
        service.get_weather.side_effect = OSError("weather API unavailable")
        tool = create_weather_tool(service)

        with self.assertRaises(RuntimeError) as context:
            tool.invoke({"location": "Tokyo"})

        self.assertIn("Weather lookup failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()