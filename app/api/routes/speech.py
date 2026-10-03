from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.routes.auth import get_current_user
from app.models.user import User
from app.schemas.speech import SpeechSynthesisRequest, TranscriptionResponse
from app.services.speech_service import InvalidSpeechTextError, SpeechSynthesisError, speech_service
from app.services.transcription_service import (
    AudioTooLargeError,
    InvalidAudioError,
    TranscriptionError,
    transcription_service,
)

router = APIRouter(prefix="/speech", tags=["Speech"])


@router.post(
    "/synthesize",
    response_class=Response,
    summary="Convert text to speech",
    responses={
        200: {
            "description": "WAV audio of the spoken text",
            "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
        },
        401: {"description": "Missing or invalid bearer token"},
        422: {"description": "Text is missing, blank, or longer than MAX_TTS_TEXT_CHARS"},
        500: {"description": "The local text-to-speech engine failed"},
    },
)
def synthesize_speech(
    request: SpeechSynthesisRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Converts text to speech with the server's local text-to-speech engine (pyttsx3 / Windows SAPI5)
    and returns it as a WAV file. No AI model or external service is called, and the audio is not stored.
    """
    try:
        audio = speech_service.synthesize(request.text)
    except InvalidSpeechTextError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except SpeechSynthesisError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe speech to text",
    responses={
        401: {"description": "Missing or invalid bearer token"},
        413: {"description": "Audio exceeds MAX_AUDIO_UPLOAD_MB"},
        422: {"description": "Missing, empty, unsupported, mislabelled, or undecodable audio"},
        500: {"description": "The local speech-to-text model failed"},
    },
)
def transcribe_speech(
    audio: UploadFile = File(..., description="WAV, MP3, M4A/MP4, WebM, OGG, or FLAC audio"),
    current_user: User = Depends(get_current_user),
):
    """
    Transcribes uploaded speech with a local Whisper model (faster-whisper on the CPU).
    No AI provider or external service is called, and the audio is not stored.

    The model is loaded on the first request, which can take noticeably longer
    (and downloads the model once if it is not already cached).
    """
    try:
        data = audio.file.read(transcription_service.max_bytes + 1)
        text = transcription_service.transcribe(data, audio.content_type)
    except AudioTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc))
    except InvalidAudioError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except TranscriptionError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return TranscriptionResponse(text=text)
