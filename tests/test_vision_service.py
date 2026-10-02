import base64
import unittest

from langchain_anthropic.chat_models import _format_messages

from app.schemas.vision import ImageTravelInsights
from app.services.vision_service import (
    VISION_INSTRUCTION,
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageTypeError,
    VisionAnalysisError,
    VisionService,
)

JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00fake-jpeg-body"
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRfake-png-body"
GIF_BYTES = b"GIF89afake-gif-body"
WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBPVP8 fake-webp-body"


def sample_insights() -> dict:
    return {
        "is_travel_related": True,
        "likely_destination": "Paris, France",
        "confidence": 0.92,
        "landmarks": ["Eiffel Tower"],
        "setting": "urban",
        "suggested_trip_style": "culture",
        "description": "The Eiffel Tower seen from the Champ de Mars on a clear day.",
    }


class FakeVisionModel:
    def __init__(self, response=None, error=None):
        self.response = sample_insights() if response is None else response
        self.error = error
        self.output_schema = None
        self.inputs = []

    def with_structured_output(self, schema):
        self.output_schema = schema
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        if self.error:
            raise self.error
        return self.response


class VisionServiceTests(unittest.TestCase):
    def assert_image_message(self, model, data, media_type):
        self.assertEqual(len(model.inputs), 1)
        [message] = model.inputs[0]
        image_block, text_block = message.content
        self.assertEqual(image_block["type"], "image")
        self.assertEqual(image_block["mime_type"], media_type)
        self.assertEqual(base64.b64decode(image_block["base64"]), data)
        self.assertEqual(text_block, {"type": "text", "text": VISION_INSTRUCTION})
        self.assertIn("travel-planning", text_block["text"])

    def test_valid_jpeg_is_sent_as_image_block(self):
        model = FakeVisionModel()

        result = VisionService(model=model).analyze(JPEG_BYTES, "image/jpeg")

        self.assertIsInstance(result, ImageTravelInsights)
        self.assertEqual(model.output_schema, ImageTravelInsights)
        self.assert_image_message(model, JPEG_BYTES, "image/jpeg")

    def test_valid_png_is_sent_as_image_block(self):
        model = FakeVisionModel()

        VisionService(model=model).analyze(PNG_BYTES, "image/png")

        self.assert_image_message(model, PNG_BYTES, "image/png")

    def test_gif_and_webp_are_supported(self):
        for data, media_type in ((GIF_BYTES, "image/gif"), (WEBP_BYTES, "image/webp")):
            model = FakeVisionModel()
            VisionService(model=model).analyze(data, media_type)
            self.assert_image_message(model, data, media_type)

    def test_message_converts_to_anthropic_base64_image_block(self):
        message = VisionService.build_message(PNG_BYTES, "image/png")

        _, formatted = _format_messages([message])

        self.assertEqual(
            formatted[0]["content"][0],
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(PNG_BYTES).decode("ascii"),
                },
            },
        )

    def test_unsupported_type_is_rejected_before_model_call(self):
        model = FakeVisionModel()

        with self.assertRaises(UnsupportedImageTypeError):
            VisionService(model=model).analyze(b"BM fake bitmap", "image/bmp")

        self.assertEqual(model.inputs, [])

    def test_oversized_image_is_rejected_before_model_call(self):
        model = FakeVisionModel()
        service = VisionService(model=model, max_bytes=len(JPEG_BYTES) - 1)

        with self.assertRaises(ImageTooLargeError):
            service.analyze(JPEG_BYTES, "image/jpeg")

        self.assertEqual(model.inputs, [])

    def test_empty_image_is_rejected_before_model_call(self):
        model = FakeVisionModel()

        with self.assertRaises(InvalidImageError):
            VisionService(model=model).analyze(b"", "image/jpeg")

        self.assertEqual(model.inputs, [])

    def test_non_image_content_is_rejected_before_model_call(self):
        model = FakeVisionModel()

        with self.assertRaises(InvalidImageError):
            VisionService(model=model).analyze(b"not really an image", "image/png")

        self.assertEqual(model.inputs, [])

    def test_content_that_does_not_match_declared_type_is_rejected(self):
        model = FakeVisionModel()

        with self.assertRaises(InvalidImageError) as context:
            VisionService(model=model).analyze(PNG_BYTES, "image/jpeg")

        self.assertIn("image/png", str(context.exception))
        self.assertEqual(model.inputs, [])

    def test_model_failure_is_wrapped(self):
        model = FakeVisionModel(error=RuntimeError("anthropic unavailable"))

        with self.assertRaises(VisionAnalysisError) as context:
            VisionService(model=model).analyze(JPEG_BYTES, "image/jpeg")

        self.assertIn("anthropic unavailable", str(context.exception))

    def test_dict_response_is_validated_into_schema(self):
        model = FakeVisionModel(response=sample_insights())

        result = VisionService(model=model).analyze(JPEG_BYTES, "image/jpeg")

        self.assertEqual(result.likely_destination, "Paris, France")
        self.assertEqual(result.landmarks, ["Eiffel Tower"])
        self.assertAlmostEqual(result.confidence, 0.92)

    def test_non_travel_response_allows_empty_destination(self):
        model = FakeVisionModel(
            response={
                "is_travel_related": False,
                "confidence": 0.0,
                "setting": "indoor",
                "description": "A screenshot of a spreadsheet.",
            }
        )

        result = VisionService(model=model).analyze(JPEG_BYTES, "image/jpeg")

        self.assertFalse(result.is_travel_related)
        self.assertIsNone(result.likely_destination)
        self.assertEqual(result.landmarks, [])

    def test_invalid_structured_response_is_rejected(self):
        model = FakeVisionModel(response={**sample_insights(), "confidence": 1.7})

        with self.assertRaises(VisionAnalysisError):
            VisionService(model=model).analyze(JPEG_BYTES, "image/jpeg")


if __name__ == "__main__":
    unittest.main()
