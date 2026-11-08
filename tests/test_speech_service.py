import io
import os
import sys
import tempfile
import threading
import time
import unittest
import wave
from unittest.mock import patch

from app.services.speech_service import (
    InvalidSpeechTextError,
    SpeechService,
    SpeechSynthesisError,
    is_wav,
)


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class FakeEngine:
    def __init__(self, behavior="ok", events=None):
        self.behavior = behavior
        self.events = events
        self.text = None
        self.path = None

    def save_to_file(self, text, filename):
        self.text = text
        self.path = filename

    def runAndWait(self):
        if self.behavior == "fail":
            raise OSError("COM error -2147200966 writing C:\\secret\\tts-file.wav")
        if self.behavior == "delete":
            os.remove(self.path)
            return
        with open(self.path, "wb") as output:
            output.write(b"not audio" if self.behavior == "garbage" else wav_bytes())

    def __del__(self):
        if self.events is not None:
            self.events.append("engine released")


class FakeEngineFactory:
    def __init__(self, behavior="ok", events=None):
        self.behavior = behavior
        self.events = events
        self.engines = []

    def __call__(self):
        if self.behavior == "create-fail":
            raise RuntimeError("SAPI5 voice not registered")
        if self.events is not None:
            self.events.append("engine created")
        engine = FakeEngine(self.behavior, self.events)
        self.engines.append(engine)
        return engine


class SpeechServiceTests(unittest.TestCase):
    def test_valid_text_returns_wav_bytes(self):
        factory = FakeEngineFactory()

        audio = SpeechService(engine_factory=factory).synthesize("  Welcome to your trip itinerary.  ")

        self.assertEqual(audio, wav_bytes())
        self.assertEqual(audio[:4], b"RIFF")
        self.assertEqual(audio[8:12], b"WAVE")
        self.assertEqual(factory.engines[0].text, "Welcome to your trip itinerary.")

    def test_engine_writes_to_temporary_wav_outside_project(self):
        factory = FakeEngineFactory()

        SpeechService(engine_factory=factory).synthesize("Hello")

        path = factory.engines[0].path
        self.assertTrue(path.endswith(".wav"))
        self.assertTrue(os.path.basename(path).startswith("tts-"))
        self.assertEqual(os.path.normcase(os.path.dirname(path)), os.path.normcase(tempfile.gettempdir()))

    def test_empty_and_whitespace_text_are_rejected_without_engine(self):
        for text in ("", "   ", "\n\t", None):
            factory = FakeEngineFactory()
            with self.assertRaises(InvalidSpeechTextError):
                SpeechService(engine_factory=factory).synthesize(text)
            self.assertEqual(factory.engines, [])

    def test_text_longer_than_limit_is_rejected_without_engine(self):
        factory = FakeEngineFactory()
        service = SpeechService(engine_factory=factory, max_chars=10)

        with self.assertRaises(InvalidSpeechTextError) as context:
            service.synthesize("x" * 11)

        self.assertIn("10-character limit", str(context.exception))
        self.assertEqual(factory.engines, [])
        self.assertEqual(service.synthesize("x" * 10), wav_bytes())

    def test_temporary_file_is_removed_after_success(self):
        factory = FakeEngineFactory()

        SpeechService(engine_factory=factory).synthesize("Hello")

        self.assertFalse(os.path.exists(factory.engines[0].path))

    def test_engine_failure_is_wrapped_and_temporary_file_removed(self):
        factory = FakeEngineFactory("fail")

        with self.assertRaises(SpeechSynthesisError) as context:
            SpeechService(engine_factory=factory).synthesize("Hello")

        self.assertEqual(str(context.exception), "Speech synthesis failed")
        self.assertFalse(os.path.exists(factory.engines[0].path))

    def test_engine_creation_failure_is_wrapped(self):
        with self.assertRaises(SpeechSynthesisError) as context:
            SpeechService(engine_factory=FakeEngineFactory("create-fail")).synthesize("Hello")

        self.assertEqual(str(context.exception), "Speech synthesis failed")

    def test_temporary_file_creation_failure_is_wrapped(self):
        factory = FakeEngineFactory()

        with patch("app.services.speech_service.tempfile.mkstemp", side_effect=OSError("disk full")):
            with self.assertRaises(SpeechSynthesisError):
                SpeechService(engine_factory=factory).synthesize("Hello")

        self.assertEqual(factory.engines, [])

    def test_missing_output_file_is_wrapped(self):
        factory = FakeEngineFactory("delete")

        with self.assertRaises(SpeechSynthesisError):
            SpeechService(engine_factory=factory).synthesize("Hello")

    def test_invalid_audio_is_rejected_and_temporary_file_removed(self):
        factory = FakeEngineFactory("garbage")

        with self.assertRaises(SpeechSynthesisError) as context:
            SpeechService(engine_factory=factory).synthesize("Hello")

        self.assertEqual(str(context.exception), "Speech synthesis produced invalid audio")
        self.assertFalse(os.path.exists(factory.engines[0].path))

    def test_each_call_gets_a_new_engine(self):
        factory = FakeEngineFactory()
        service = SpeechService(engine_factory=factory)

        service.synthesize("First")
        service.synthesize("Second")

        self.assertEqual(len(factory.engines), 2)
        self.assertIsNot(factory.engines[0], factory.engines[1])
        self.assertNotEqual(factory.engines[0].path, factory.engines[1].path)

    def test_concurrent_calls_never_use_engines_at_the_same_time(self):
        state = {"active": 0, "max_active": 0}
        guard = threading.Lock()

        class SlowEngine(FakeEngine):
            def runAndWait(self):
                with guard:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                time.sleep(0.05)
                super().runAndWait()
                with guard:
                    state["active"] -= 1

        service = SpeechService(engine_factory=SlowEngine)
        results = []
        threads = [threading.Thread(target=lambda: results.append(service.synthesize("Hello"))) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results, [wav_bytes()] * 4)
        self.assertEqual(state["max_active"], 1)

    def test_lock_is_released_after_failure(self):
        with self.assertRaises(SpeechSynthesisError):
            SpeechService(engine_factory=FakeEngineFactory("fail")).synthesize("Hello")

        self.assertEqual(SpeechService(engine_factory=FakeEngineFactory()).synthesize("Hello"), wav_bytes())

    @unittest.skipUnless(sys.platform == "win32", "COM is Windows-only")
    def test_engine_is_released_inside_com_scope(self):
        events = []

        with patch("pythoncom.CoInitialize", side_effect=lambda: events.append("com initialized")), patch(
            "pythoncom.CoUninitialize", side_effect=lambda: events.append("com uninitialized")
        ):
            def factory():
                events.append("engine created")
                return FakeEngine(events=events)

            SpeechService(engine_factory=factory).synthesize("Hello")

        self.assertEqual(events, ["com initialized", "engine created", "engine released", "com uninitialized"])

    def test_is_wav(self):
        self.assertTrue(is_wav(wav_bytes()))
        self.assertFalse(is_wav(b"RIFF"))
        self.assertFalse(is_wav(b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 64))


if __name__ == "__main__":
    unittest.main()
