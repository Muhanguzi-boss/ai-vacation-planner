import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.routes import itineraries
from app.api.routes.auth import get_current_user
from app.db.database import get_db
from app.services.speech_service import SpeechService


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


SAVED_ITINERARY = SimpleNamespace(
    trip=SimpleNamespace(destination="Paris"),
    days=[
        {
            "day": 1,
            "activities": [
                {"time": "morning", "activity": "Visit the Louvre", "location": "Louvre, Paris", "estimated_cost": "$22"}
            ],
        }
    ],
    total_estimated_cost="$22",
)


class FakeEngine:
    def __init__(self, spoken, fail):
        self.spoken = spoken
        self.fail = fail

    def save_to_file(self, text, filename):
        self.spoken.append(text)
        self.path = filename

    def runAndWait(self):
        if self.fail:
            raise OSError("COM error at C:\\Users\\secret\\tts.wav")
        with open(self.path, "wb") as output:
            output.write(wav_bytes())


def build_app(authenticated=True):
    app = FastAPI()
    app.include_router(itineraries.router)
    app.dependency_overrides[get_db] = lambda: None
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7, username="alice")
    return app


class ItineraryAudioRouteTests(unittest.TestCase):
    def setUp(self):
        self.spoken = []
        self.lookups = []
        self.tts_fails = False
        self.lookup_result = SAVED_ITINERARY
        service = SpeechService(engine_factory=lambda: FakeEngine(self.spoken, self.tts_fails))

        def lookup(db, trip_id, user_id):
            self.lookups.append((trip_id, user_id))
            if isinstance(self.lookup_result, Exception):
                raise self.lookup_result
            return self.lookup_result

        for name, value in (("speech_service", service), ("get_itinerary_by_trip", lookup)):
            patcher = patch.object(itineraries, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def post(self, authenticated=True):
        return TestClient(build_app(authenticated)).post("/itineraries/3/audio")

    def test_requires_authentication(self):
        response = self.post(authenticated=False)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.spoken, [])

    def test_returns_wav_narration_of_saved_itinerary(self):
        response = self.post()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav_bytes())
        self.assertEqual(self.lookups, [(3, 7)])
        self.assertEqual(
            self.spoken,
            ["Here is your 1-day itinerary for Paris. Day 1. Morning: Visit the Louvre. Estimated total cost: $22."],
        )

    def test_missing_itinerary_returns_404_without_synthesis(self):
        self.lookup_result = HTTPException(status_code=404, detail="Itinerary not found")

        response = self.post()

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Itinerary not found"})
        self.assertEqual(self.spoken, [])

    def test_tts_failure_returns_generic_500(self):
        self.tts_fails = True

        response = self.post()

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Speech synthesis failed"})
        self.assertNotIn("secret", response.text)

    def test_openapi_documents_wav_response(self):
        operation = build_app().openapi()["paths"]["/itineraries/{trip_id}/audio"]["post"]

        self.assertIn("audio/wav", operation["responses"]["200"]["content"])
        self.assertEqual(operation["security"], [{"OAuth2PasswordBearer": []}])
        self.assertTrue({"401", "404", "500"} <= set(operation["responses"]))


if __name__ == "__main__":
    unittest.main()
