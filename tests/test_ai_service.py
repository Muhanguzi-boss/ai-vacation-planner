import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.ai_service import AIService

VALID_ITINERARY = {
    "destination": "Paris",
    "days": [
        {
            "day": 1,
            "activities": [
                {"time": "morning", "activity": "Louvre", "location": "Paris", "estimated_cost": "$22"}
            ],
        }
    ],
    "total_estimated_cost": "$22",
}


class FakeAnthropicClient:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def service_with(client, model=None):
    environment = {"ANTHROPIC_MODEL": model} if model else {}
    with patch.dict("os.environ", environment, clear=False):
        service = AIService()
    service.anthropic_client = client
    return service


class AIServiceProviderTests(unittest.TestCase):
    def test_anthropic_is_the_only_llm_provider(self):
        service = AIService()

        for attribute in ("openai_client", "openai_key", "openai_model", "default_provider"):
            self.assertFalse(hasattr(service, attribute), attribute)

    def test_default_model_is_current(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(AIService().anthropic_model, "claude-haiku-4-5")

    def test_generate_itinerary_uses_anthropic_and_validates_output(self):
        client = FakeAnthropicClient(text="```json\n" + json.dumps(VALID_ITINERARY) + "\n```")

        result = service_with(client, model="claude-haiku-4-5").generate_itinerary("Paris", 1, 100.0, "culture")

        self.assertEqual(result, VALID_ITINERARY)
        self.assertEqual(client.requests[0]["model"], "claude-haiku-4-5")

    def test_anthropic_failure_uses_mock_fallback(self):
        client = FakeAnthropicClient(error=RuntimeError("anthropic unavailable"))

        result = service_with(client).generate_itinerary("Rome", 2, None, None)

        self.assertEqual(result["destination"], "Rome")
        self.assertEqual(len(result["days"]), 2)

    def test_invalid_model_output_uses_mock_fallback(self):
        result = service_with(FakeAnthropicClient(text="not json")).generate_itinerary("Rome", 1, None, None)

        self.assertEqual(result["total_estimated_cost"], "$45")

    def test_missing_anthropic_client_uses_mock_fallback(self):
        result = service_with(None).generate_itinerary("Tokyo", 1, None, None)

        self.assertEqual(result["destination"], "Tokyo")


if __name__ == "__main__":
    unittest.main()
