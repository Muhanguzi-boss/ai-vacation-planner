import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.ai_service import AIService


class FakeOpenAIClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        content = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class AIServiceTests(unittest.TestCase):
    def _make_service(self, responses=None):
        service = AIService.__new__(AIService)
        service.openai_key = "test"
        service.anthropic_key = None
        service.default_provider = "openai"
        service.temperature = 0.1
        service.max_tokens = 1000
        service.openai_model = "gpt-4o-mini"
        service.anthropic_model = "claude-3-5-sonnet"
        service.openai_client = FakeOpenAIClient(responses or [])
        service.anthropic_client = None
        service.max_retries = 2
        service.retry_backoff_seconds = 0.0
        return service

    def test_valid_json_response_is_validated_and_normalized(self):
        service = self._make_service(
            responses=[
                '{"trip_id": 7, "itinerary": [{"day": 1, "activities": ["Visit the Eiffel Tower", "Walk along the Seine"]}]}'
            ]
        )

        with patch.object(service, "_generate_with_openai", return_value='{"trip_id": 7, "itinerary": [{"day": 1, "activities": ["Visit the Eiffel Tower", "Walk along the Seine"]}]}'):
            result = service.generate_itinerary("Paris", 1, 250.0, "romantic", trip_id=7)

        self.assertEqual(result["trip_id"], 7)
        self.assertEqual(result["itinerary"][0]["day"], 1)
        self.assertEqual(result["itinerary"][0]["activities"][0], "Visit the Eiffel Tower")

    def test_invalid_json_response_falls_back_to_mock(self):
        service = self._make_service(responses=["not valid json"])

        with patch.object(service, "_generate_with_openai", return_value="not valid json"):
            with patch("app.services.ai_service.time.sleep", return_value=None):
                result = service.generate_itinerary("Rome", 2, 120.0, "budget", trip_id=9)

        self.assertEqual(result["trip_id"], 9)
        self.assertEqual(result["itinerary"][0]["day"], 1)
        self.assertGreaterEqual(len(result["itinerary"]), 1)


if __name__ == "__main__":
    unittest.main()
