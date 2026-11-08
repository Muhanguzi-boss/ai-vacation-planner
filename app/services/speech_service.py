"""Local text-to-speech using pyttsx3 (SAPI5 on Windows). Makes no LLM or network calls."""

import logging
import os
import sys
import tempfile
import threading
from contextlib import contextmanager, suppress
from typing import Any, Callable, Iterator

from app.core.config import settings

logger = logging.getLogger(__name__)

_synthesis_lock = threading.Lock()


class InvalidSpeechTextError(ValueError):
    """Raised when text cannot be synthesized."""


class SpeechSynthesisError(RuntimeError):
    """Raised when the TTS engine fails to produce audio."""


def create_pyttsx3_engine() -> Any:
    import pyttsx3

    return pyttsx3.Engine()


@contextmanager
def _com_initialized() -> Iterator[None]:
    if sys.platform != "win32":
        yield
        return
    import pythoncom

    pythoncom.CoInitialize()
    try:
        yield
    finally:
        pythoncom.CoUninitialize()


def is_wav(data: bytes) -> bool:
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


class SpeechService:
    def __init__(
        self,
        engine_factory: Callable[[], Any] | None = None,
        max_chars: int | None = None,
    ) -> None:
        self.engine_factory = engine_factory or create_pyttsx3_engine
        self.max_chars = max_chars or settings.MAX_TTS_TEXT_CHARS

    def validate_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise InvalidSpeechTextError("Text must not be empty")
        text = text.strip()
        if len(text) > self.max_chars:
            raise InvalidSpeechTextError(f"Text exceeds the {self.max_chars}-character limit")
        return text

    def synthesize(self, text: str) -> bytes:
        text = self.validate_text(text)
        path = None
        with _synthesis_lock:
            try:
                handle, path = tempfile.mkstemp(prefix="tts-", suffix=".wav")
                os.close(handle)
                with _com_initialized():
                    engine = self.engine_factory()
                    try:
                        engine.save_to_file(text, path)
                        engine.runAndWait()
                    finally:
                        del engine
                with open(path, "rb") as audio_file:
                    audio = audio_file.read()
            except Exception as exc:
                logger.exception("Text-to-speech synthesis failed")
                raise SpeechSynthesisError("Speech synthesis failed") from exc
            finally:
                if path:
                    with suppress(OSError):
                        os.remove(path)

        if not is_wav(audio):
            logger.error("Text-to-speech engine produced %d bytes of invalid WAV data", len(audio))
            raise SpeechSynthesisError("Speech synthesis produced invalid audio")
        return audio


speech_service = SpeechService()
