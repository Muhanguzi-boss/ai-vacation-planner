"""Local speech-to-text using faster-whisper on the CPU. Makes no LLM calls.

The Whisper model is loaded once, on first use, and reused for every request. The first load
downloads the model into the Hugging Face cache if it is not already there.
"""

import logging
import os
import tempfile
import threading
from contextlib import suppress
from typing import Any, Callable

from app.core.config import settings

logger = logging.getLogger(__name__)

AUDIO_FORMATS = {
    "wav": ({"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}, ".wav"),
    "mp3": ({"audio/mpeg", "audio/mp3"}, ".mp3"),
    "mp4": ({"audio/mp4", "audio/x-m4a", "audio/m4a", "video/mp4"}, ".m4a"),
    "webm": ({"audio/webm", "video/webm"}, ".webm"),
    "ogg": ({"audio/ogg", "application/ogg"}, ".ogg"),
    "flac": ({"audio/flac", "audio/x-flac"}, ".flac"),
}
GENERIC_CONTENT_TYPE = "application/octet-stream"
SUPPORTED_CONTENT_TYPES = {GENERIC_CONTENT_TYPE}.union(*(types for types, _ in AUDIO_FORMATS.values()))


class InvalidAudioError(ValueError):
    """Raised when uploaded audio is missing, unsupported, or cannot be decoded."""


class AudioTooLargeError(InvalidAudioError):
    """Raised when uploaded audio exceeds the configured size limit."""


class TranscriptionError(RuntimeError):
    """Raised when the speech-to-text model fails."""


def detect_audio_format(data: bytes) -> str | None:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0 and data[1] & 0x06):
        return "mp3"
    if data[4:8] == b"ftyp":
        return "mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    return None


def create_whisper_model() -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(settings.WHISPER_MODEL, device="cpu", compute_type=settings.WHISPER_COMPUTE_TYPE)


def _is_decode_error(exc: Exception) -> bool:
    from av.error import FFmpegError

    return isinstance(exc, FFmpegError)


class TranscriptionService:
    def __init__(
        self,
        model_factory: Callable[[], Any] | None = None,
        max_bytes: int | None = None,
        max_duration_seconds: float | None = None,
    ) -> None:
        self.model_factory = model_factory or create_whisper_model
        self.max_bytes = max_bytes or settings.MAX_AUDIO_UPLOAD_MB * 1024 * 1024
        self.max_duration_seconds = max_duration_seconds or settings.MAX_AUDIO_DURATION_SECONDS
        self._model = None
        self._lock = threading.Lock()

    def validate_audio(self, data: bytes, content_type: str | None) -> str:
        media_type = (content_type or "").split(";")[0].strip().lower()
        if media_type not in SUPPORTED_CONTENT_TYPES:
            raise InvalidAudioError(f"Unsupported audio content type '{media_type}'")
        if not data:
            raise InvalidAudioError("Audio file is empty")
        if len(data) > self.max_bytes:
            raise AudioTooLargeError(f"Audio exceeds the {self.max_bytes // (1024 * 1024)} MB upload limit")
        audio_format = detect_audio_format(data)
        if audio_format is None:
            raise InvalidAudioError("Unsupported audio format. Supported formats: WAV, MP3, M4A/MP4, WebM, OGG, FLAC")
        content_types, suffix = AUDIO_FORMATS[audio_format]
        if media_type != GENERIC_CONTENT_TYPE and media_type not in content_types:
            raise InvalidAudioError(f"File content is {audio_format.upper()} audio but was declared as {media_type}")
        return suffix

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self.model_factory()
        return self._model

    def transcribe(self, data: bytes, content_type: str | None) -> str:
        suffix = self.validate_audio(data, content_type)
        path = None
        try:
            handle, path = tempfile.mkstemp(prefix="stt-", suffix=suffix)
            with os.fdopen(handle, "wb") as audio_file:
                audio_file.write(data)
            with self._lock:
                segments, info = self._get_model().transcribe(path)
                if info.duration > self.max_duration_seconds:
                    raise InvalidAudioError(f"Audio exceeds the {self.max_duration_seconds:g}-second duration limit")
                return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        except InvalidAudioError:
            raise
        except Exception as exc:
            if _is_decode_error(exc):
                raise InvalidAudioError("Audio could not be decoded") from exc
            logger.exception("Speech-to-text transcription failed")
            raise TranscriptionError("Transcription failed") from exc
        finally:
            if path:
                with suppress(OSError):
                    os.remove(path)


transcription_service = TranscriptionService()
