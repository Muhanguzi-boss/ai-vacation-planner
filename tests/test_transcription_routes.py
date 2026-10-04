import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import speech
from app.api.routes.auth import get_current_user
from app.services.transcription_service import TranscriptionService


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class FakeWhisperModel:
    def __init__(self):
        self.calls = 0
        self.error = None

    def transcribe(self, path):
        self.calls += 1
        if self.error:
            raise self.error
        return iter([SimpleNamespace(text=" Plan a trip to Paris.")]), SimpleNamespace(language="en", duration=2.0)


def build_app(authenticated=True):
    app = FastAPI()
    app.include_router(speech.router)
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, username="alice")
    return app


class TranscriptionRouteTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeWhisperModel()
        service = TranscriptionService(model_factory=lambda: self.model, max_bytes=10_000)
        patcher = patch.object(speech, "transcription_service", service)
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, data=None, content_type="audio/wav", authenticated=True):
        files = None if data is None else {"audio": ("recording", data, content_type)}
        return TestClient(build_app(authenticated)).post("/speech/transcribe", files=files)

    def test_requires_authentication(self):
        response = self.post(wav_bytes(), authenticated=False)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.model.calls, 0)

    def test_returns_transcription(self):
        response = self.post(wav_bytes())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "Plan a trip to Paris."})

    def test_missing_file_returns_422(self):
        self.assertEqual(self.post().status_code, 422)

    def test_unsupported_content_type_returns_422(self):
        response = self.post(wav_bytes(), content_type="text/plain")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.model.calls, 0)

    def test_unsupported_format_returns_422(self):
        response = self.post(b"definitely not audio", content_type="audio/wav")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.model.calls, 0)

    def test_empty_audio_returns_422(self):
        response = self.post(b"")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "Audio file is empty"})

    def test_oversized_audio_returns_413(self):
        response = self.post(wav_bytes() + b"\x00" * 20_000)

        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.model.calls, 0)

    def test_over_long_audio_returns_422(self):
        long_model = FakeWhisperModel()
        long_model.transcribe = lambda path: (iter([]), SimpleNamespace(language="en", duration=10_000.0))
        service = TranscriptionService(model_factory=lambda: long_model, max_bytes=10_000)

        with patch.object(speech, "transcription_service", service):
            response = self.post(wav_bytes())

        self.assertEqual(response.status_code, 422)
        self.assertIn("duration limit", response.json()["detail"])

    def test_model_failure_returns_generic_500(self):
        self.model.error = RuntimeError("model file C:\\Users\\secret\\model.bin is corrupt")

        response = self.post(wav_bytes())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Transcription failed"})
        self.assertNotIn("secret", response.text)

    def test_openapi_documents_transcription_and_keeps_synthesis(self):
        schema = build_app().openapi()
        operation = schema["paths"]["/speech/transcribe"]["post"]

        self.assertEqual(operation["tags"], ["Speech"])
        self.assertEqual(operation["security"], [{"OAuth2PasswordBearer": []}])
        self.assertIn("multipart/form-data", operation["requestBody"]["content"])
        self.assertEqual(
            operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/TranscriptionResponse",
        )
        self.assertTrue({"401", "413", "422", "500"} <= set(operation["responses"]))
        self.assertIn("/speech/synthesize", schema["paths"])


if __name__ == "__main__":
    unittest.main()
