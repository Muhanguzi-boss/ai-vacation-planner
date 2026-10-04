import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import speech
from app.api.routes.auth import get_current_user
from app.services.ai_service import GeneratedItinerary
from app.services.transcription_service import TranscriptionService


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


ITINERARY = GeneratedItinerary.model_validate(
    {
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
)


class FakeWhisperModel:
    def __init__(self):
        self.texts = [" Plan a day in Paris with museums."]

    def transcribe(self, path):
        segments = (SimpleNamespace(text=text) for text in self.texts)
        return segments, SimpleNamespace(language="en", duration=3.0)


def build_app(authenticated=True):
    app = FastAPI()
    app.include_router(speech.router)
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, username="alice")
    return app


class VoicePlanRouteTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeWhisperModel()
        self.orchestrator_class = MagicMock()
        self.orchestrator_class.return_value.invoke.return_value = ITINERARY
        service = TranscriptionService(model_factory=lambda: self.model)
        for name, value in (("transcription_service", service), ("TravelPlanningOrchestrator", self.orchestrator_class)):
            patcher = patch.object(speech, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def post(self, data=None, content_type="audio/wav", authenticated=True):
        data = wav_bytes() if data is None else data
        return TestClient(build_app(authenticated)).post(
            "/speech/plan", files={"audio": ("request.wav", data, content_type)}
        )

    def test_requires_authentication(self):
        response = self.post(authenticated=False)

        self.assertEqual(response.status_code, 401)
        self.orchestrator_class.assert_not_called()

    def test_transcript_is_planned_by_existing_orchestrator(self):
        response = self.post()

        self.assertEqual(response.status_code, 200)
        self.orchestrator_class.return_value.invoke.assert_called_once_with("Plan a day in Paris with museums.")
        self.assertEqual(
            response.json(),
            {"transcript": "Plan a day in Paris with museums.", **ITINERARY.model_dump()},
        )

    def test_no_speech_returns_422_without_planning(self):
        self.model.texts = []

        response = self.post()

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "No speech was detected in the audio"})
        self.orchestrator_class.assert_not_called()

    def test_invalid_audio_returns_422_without_planning(self):
        response = self.post(b"not audio at all")

        self.assertEqual(response.status_code, 422)
        self.orchestrator_class.assert_not_called()

    def test_planning_failure_returns_generic_502(self):
        self.orchestrator_class.return_value.invoke.side_effect = RuntimeError("Weather lookup failed: secret detail")

        response = self.post()

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "AI generation failed"})
        self.assertNotIn("secret", response.text)

    def test_missing_anthropic_configuration_returns_generic_502(self):
        self.orchestrator_class.side_effect = ValueError("ANTHROPIC_API_KEY is required for travel orchestration")

        response = self.post()

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("ANTHROPIC_API_KEY", response.text)


if __name__ == "__main__":
    unittest.main()
