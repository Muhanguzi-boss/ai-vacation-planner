import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import vision
from app.api.routes.auth import get_current_user
from app.services.vision_service import VisionService

JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00fake-jpeg-body"
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRfake-png-body"


class FakeVisionModel:
    def __init__(self):
        self.error = None
        self.inputs = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        if self.error:
            raise self.error
        return {
            "is_travel_related": True,
            "likely_destination": "Paris, France",
            "confidence": 0.92,
            "landmarks": ["Eiffel Tower"],
            "setting": "urban",
            "suggested_trip_style": "culture",
            "description": "The Eiffel Tower seen from the Champ de Mars on a clear day.",
        }


def build_client(authenticated=True):
    app = FastAPI()
    app.include_router(vision.router)
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, username="alice")
    return TestClient(app)


class VisionRouteTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeVisionModel()
        self.service = VisionService(model=self.model, max_bytes=1024)
        patcher = patch.object(vision, "vision_service", self.service)
        patcher.start()
        self.addCleanup(patcher.stop)

    def post_image(self, data, content_type, client=None):
        client = client or build_client()
        return client.post("/vision/analyze", files={"image": ("upload", data, content_type)})

    def test_requires_authentication(self):
        response = self.post_image(JPEG_BYTES, "image/jpeg", client=build_client(authenticated=False))

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.model.inputs, [])

    def test_valid_jpeg_returns_structured_insights(self):
        response = self.post_image(JPEG_BYTES, "image/jpeg")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["likely_destination"], "Paris, France")
        self.assertEqual(body["landmarks"], ["Eiffel Tower"])
        self.assertTrue(body["is_travel_related"])

    def test_valid_png_is_accepted(self):
        response = self.post_image(PNG_BYTES, "image/png")

        self.assertEqual(response.status_code, 200)
        [message] = self.model.inputs[0]
        self.assertEqual(message.content[0]["mime_type"], "image/png")

    def test_unsupported_type_returns_415(self):
        response = self.post_image(b"%PDF-1.7", "application/pdf")

        self.assertEqual(response.status_code, 415)
        self.assertEqual(self.model.inputs, [])

    def test_oversized_image_returns_413(self):
        response = self.post_image(JPEG_BYTES + b"\x00" * 2048, "image/jpeg")

        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.model.inputs, [])

    def test_empty_file_returns_400(self):
        response = self.post_image(b"", "image/jpeg")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.model.inputs, [])

    def test_invalid_image_content_returns_400(self):
        response = self.post_image(b"plain text pretending to be a png", "image/png")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.model.inputs, [])

    def test_model_failure_returns_502(self):
        self.model.error = RuntimeError("anthropic unavailable")

        response = self.post_image(JPEG_BYTES, "image/jpeg")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "AI image analysis failed"})
        self.assertNotIn("anthropic unavailable", response.text)

    def test_missing_file_returns_422(self):
        response = build_client().post("/vision/analyze")

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
