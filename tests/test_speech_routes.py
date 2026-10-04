import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import speech
from app.api.routes.auth import get_current_user
from app.core.config import settings
from app.services.speech_service import SpeechService


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


class FakeEngine:
    def __init__(self, calls, behavior):
        self.calls = calls
        self.behavior = behavior

    def save_to_file(self, text, filename):
        self.calls.append(text)
        self.path = filename

    def runAndWait(self):
        if self.behavior == "fail":
            raise OSError("COM error -2147200966 at C:\\Users\\secret\\tts-file.wav")
        with open(self.path, "wb") as output:
            output.write(b"garbage" if self.behavior == "garbage" else wav_bytes())


def build_app(authenticated=True):
    app = FastAPI()
    app.include_router(speech.router)
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, username="alice")
    return app


class SpeechRouteTests(unittest.TestCase):
    def use_engine(self, behavior="ok", max_chars=None):
        self.calls = []
        service = SpeechService(engine_factory=lambda: FakeEngine(self.calls, behavior), max_chars=max_chars)
        patcher = patch.object(speech, "speech_service", service)
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, payload, authenticated=True):
        return TestClient(build_app(authenticated)).post("/speech/synthesize", json=payload)

    def test_requires_authentication(self):
        self.use_engine()

        response = self.post({"text": "Hello"}, authenticated=False)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.calls, [])

    def test_returns_wav_audio(self):
        self.use_engine()

        response = self.post({"text": "Welcome to your trip itinerary."})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav_bytes())
        self.assertEqual(response.content[:4], b"RIFF")
        self.assertEqual(self.calls, ["Welcome to your trip itinerary."])

    def test_invalid_text_returns_422_without_engine(self):
        self.use_engine()

        for payload in ({"text": ""}, {"text": "   "}, {"text": "x" * (settings.MAX_TTS_TEXT_CHARS + 1)}, {}, {"text": 42}):
            response = self.post(payload)
            self.assertEqual(response.status_code, 422, payload)
        self.assertEqual(self.calls, [])

    def test_service_text_limit_returns_422(self):
        self.use_engine(max_chars=5)

        response = self.post({"text": "longer than five"})

        self.assertEqual(response.status_code, 422)
        self.assertIn("5-character limit", response.json()["detail"])

    def test_engine_failure_returns_500_without_internal_details(self):
        self.use_engine("fail")

        response = self.post({"text": "Hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Speech synthesis failed"})
        self.assertNotIn("secret", response.text)

    def test_invalid_generated_audio_returns_500(self):
        self.use_engine("garbage")

        response = self.post({"text": "Hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Speech synthesis produced invalid audio"})

    def test_openapi_documents_wav_response_and_request_schema(self):
        operation = build_app().openapi()["paths"]["/speech/synthesize"]["post"]

        self.assertEqual(operation["tags"], ["Speech"])
        self.assertIn("audio/wav", operation["responses"]["200"]["content"])
        self.assertNotIn("application/json", operation["responses"]["200"]["content"])
        self.assertEqual(
            operation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/SpeechSynthesisRequest",
        )
        self.assertTrue({"401", "422", "500"} <= set(operation["responses"]))


if __name__ == "__main__":
    unittest.main()
