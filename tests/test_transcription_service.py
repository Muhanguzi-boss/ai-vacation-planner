import io
import os
import tempfile
import threading
import time
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

from av.error import InvalidDataError

from app.services.transcription_service import (
    AudioTooLargeError,
    InvalidAudioError,
    TranscriptionError,
    TranscriptionService,
    detect_audio_format,
)


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


SAMPLES = {
    "wav": (wav_bytes(), "audio/wav", ".wav"),
    "mp3-id3": (b"ID3\x04\x00\x00" + b"\x00" * 64, "audio/mpeg", ".mp3"),
    "mp3-frame": (b"\xff\xfb\x90\x00" + b"\x00" * 64, "audio/mpeg", ".mp3"),
    "m4a": (b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 64, "audio/x-m4a", ".m4a"),
    "mp4": (b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, "video/mp4", ".m4a"),
    "webm": (b"\x1a\x45\xdf\xa3" + b"\x00" * 64, "audio/webm;codecs=opus", ".webm"),
    "ogg": (b"OggS" + b"\x00" * 64, "audio/ogg", ".ogg"),
    "flac": (b"fLaC" + b"\x00" * 64, "audio/flac", ".flac"),
}


class FakeWhisperModel:
    def __init__(self, texts=(" Hello", " world. "), error=None):
        self.texts = texts
        self.error = error
        self.calls = []

    def transcribe(self, path):
        with open(path, "rb") as audio_file:
            self.calls.append({"path": path, "data": audio_file.read()})
        if self.error:
            raise self.error
        segments = (SimpleNamespace(text=text) for text in self.texts)
        return segments, SimpleNamespace(language="en", duration=1.0)


class CountingFactory:
    def __init__(self, model=None, error=None, delay=0):
        self.model = model or FakeWhisperModel()
        self.error = error
        self.delay = delay
        self.calls = 0

    def __call__(self):
        self.calls += 1
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.model


class TranscriptionServiceTests(unittest.TestCase):
    def service(self, factory=None, max_bytes=None):
        self.factory = factory or CountingFactory()
        return TranscriptionService(model_factory=self.factory, max_bytes=max_bytes)

    def test_wav_is_transcribed(self):
        text = self.service().transcribe(wav_bytes(), "audio/wav")

        self.assertEqual(text, "Hello world.")
        [call] = self.factory.model.calls
        self.assertEqual(call["data"], wav_bytes())
        self.assertTrue(call["path"].endswith(".wav"))
        self.assertEqual(os.path.normcase(os.path.dirname(call["path"])), os.path.normcase(tempfile.gettempdir()))

    def test_supported_formats_are_detected_and_written_with_matching_suffix(self):
        for name, (data, content_type, suffix) in SAMPLES.items():
            service = self.service()
            service.transcribe(data, content_type)
            [call] = self.factory.model.calls
            self.assertTrue(call["path"].endswith(suffix), name)
            self.assertEqual(call["data"], data, name)

    def test_generic_content_type_is_accepted_when_content_is_audio(self):
        self.assertEqual(self.service().transcribe(wav_bytes(), "application/octet-stream"), "Hello world.")

    def test_temporary_file_is_removed_after_success(self):
        service = self.service()

        service.transcribe(wav_bytes(), "audio/wav")

        self.assertFalse(os.path.exists(self.factory.model.calls[0]["path"]))

    def test_unsupported_content_type_is_rejected_before_model_load(self):
        with self.assertRaises(InvalidAudioError) as context:
            self.service().transcribe(wav_bytes(), "image/png")

        self.assertIn("Unsupported audio content type", str(context.exception))
        self.assertEqual(self.factory.calls, 0)

    def test_missing_content_type_is_rejected(self):
        with self.assertRaises(InvalidAudioError):
            self.service().transcribe(wav_bytes(), None)

    def test_unrecognized_content_is_rejected_before_model_load(self):
        for data in (b"plain text pretending to be audio", b"\xff\xf1\x50\x80" + b"\x00" * 32):
            with self.assertRaises(InvalidAudioError):
                self.service().transcribe(data, "audio/mpeg")
            self.assertEqual(self.factory.calls, 0)

    def test_content_that_does_not_match_declared_type_is_rejected(self):
        with self.assertRaises(InvalidAudioError) as context:
            self.service().transcribe(SAMPLES["mp3-id3"][0], "audio/wav")

        self.assertIn("MP3", str(context.exception))
        self.assertEqual(self.factory.calls, 0)

    def test_empty_audio_is_rejected(self):
        with self.assertRaises(InvalidAudioError) as context:
            self.service().transcribe(b"", "audio/wav")

        self.assertEqual(str(context.exception), "Audio file is empty")
        self.assertEqual(self.factory.calls, 0)

    def test_oversized_audio_is_rejected(self):
        with self.assertRaises(AudioTooLargeError):
            self.service(max_bytes=100).transcribe(wav_bytes(), "audio/wav")

        self.assertEqual(self.factory.calls, 0)

    def test_undecodable_audio_is_invalid_input_and_cleaned_up(self):
        model = FakeWhisperModel(error=InvalidDataError(1094995529, "Invalid data found when processing input"))

        with self.assertRaises(InvalidAudioError) as context:
            self.service(CountingFactory(model)).transcribe(wav_bytes(), "audio/wav")

        self.assertEqual(str(context.exception), "Audio could not be decoded")
        self.assertFalse(os.path.exists(model.calls[0]["path"]))

    def test_model_failure_is_generic_and_cleaned_up(self):
        model = FakeWhisperModel(error=RuntimeError("ctranslate2 failure at C:\\models\\secret"))

        with self.assertRaises(TranscriptionError) as context:
            self.service(CountingFactory(model)).transcribe(wav_bytes(), "audio/wav")

        self.assertEqual(str(context.exception), "Transcription failed")
        self.assertFalse(os.path.exists(model.calls[0]["path"]))

    def test_temporary_file_failure_is_generic(self):
        with patch("app.services.transcription_service.tempfile.mkstemp", side_effect=OSError("disk full")):
            with self.assertRaises(TranscriptionError):
                self.service().transcribe(wav_bytes(), "audio/wav")

    def test_model_is_loaded_once_and_reused(self):
        service = self.service()

        service.transcribe(wav_bytes(), "audio/wav")
        service.transcribe(wav_bytes(), "audio/wav")

        self.assertEqual(self.factory.calls, 1)
        self.assertEqual(len(self.factory.model.calls), 2)

    def test_failed_model_load_is_retried_on_next_request(self):
        factory = CountingFactory(error=OSError("model download failed"))
        service = self.service(factory)

        with self.assertRaises(TranscriptionError):
            service.transcribe(wav_bytes(), "audio/wav")
        factory.error = None

        self.assertEqual(service.transcribe(wav_bytes(), "audio/wav"), "Hello world.")
        self.assertEqual(factory.calls, 2)

    def test_concurrent_requests_load_model_once_and_never_overlap(self):
        state = {"active": 0, "max_active": 0}
        guard = threading.Lock()

        class SlowModel(FakeWhisperModel):
            def transcribe(self, path):
                with guard:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                time.sleep(0.05)
                result = super().transcribe(path)
                with guard:
                    state["active"] -= 1
                return result

        service = self.service(CountingFactory(SlowModel(), delay=0.1))
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(service.transcribe(wav_bytes(), "audio/wav")))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results, ["Hello world."] * 4)
        self.assertEqual(self.factory.calls, 1)
        self.assertEqual(state["max_active"], 1)

    def test_audio_longer_than_limit_is_rejected_before_inference(self):
        consumed = []

        class LongAudioModel(FakeWhisperModel):
            def transcribe(self, path):
                super().transcribe(path)
                segments = (consumed.append(text) or SimpleNamespace(text=text) for text in self.texts)
                return segments, SimpleNamespace(language="en", duration=600.0)

        model = LongAudioModel()
        service = TranscriptionService(model_factory=lambda: model, max_duration_seconds=300)

        with self.assertRaises(InvalidAudioError) as context:
            service.transcribe(wav_bytes(), "audio/wav")

        self.assertEqual(str(context.exception), "Audio exceeds the 300-second duration limit")
        self.assertEqual(consumed, [])
        self.assertFalse(os.path.exists(model.calls[0]["path"]))

    def test_duration_limit_defaults_to_settings(self):
        with patch("app.services.transcription_service.settings", SimpleNamespace(
            MAX_AUDIO_UPLOAD_MB=10, MAX_AUDIO_DURATION_SECONDS=42
        )):
            self.assertEqual(TranscriptionService().max_duration_seconds, 42)

    def test_audio_without_speech_returns_empty_text(self):
        self.assertEqual(self.service(CountingFactory(FakeWhisperModel(texts=()))).transcribe(wav_bytes(), "audio/wav"), "")

    def test_detect_audio_format(self):
        self.assertEqual(detect_audio_format(wav_bytes()), "wav")
        self.assertEqual(detect_audio_format(b"RIFF\x00\x00\x00\x00AVI "), None)
        self.assertEqual(detect_audio_format(b""), None)


if __name__ == "__main__":
    unittest.main()
