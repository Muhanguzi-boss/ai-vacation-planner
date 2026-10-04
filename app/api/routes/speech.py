import logging

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.routes.auth import get_current_user
from app.models.user import User
from app.schemas.speech import SpeechSynthesisRequest, TranscriptionResponse, VoicePlanResponse
from app.services.agent_service import TravelPlanningOrchestrator
from app.services.speech_service import InvalidSpeechTextError, SpeechSynthesisError, speech_service
from app.services.transcription_service import (
    AudioTooLargeError,
    InvalidAudioError,
    TranscriptionError,
    transcription_service,
)

logger = logging.getLogger(__name__)

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
        422: {"description": "Missing, empty, unsupported, mislabelled, undecodable, or over-long audio"},
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
    return TranscriptionResponse(text=_transcribe_upload(audio))


@router.post(
    "/plan",
    response_model=VoicePlanResponse,
    summary="Plan an itinerary from a spoken request",
    responses={
        401: {"description": "Missing or invalid bearer token"},
        413: {"description": "Audio exceeds MAX_AUDIO_UPLOAD_MB"},
        422: {"description": "Invalid audio, or no speech was detected"},
        500: {"description": "The local speech-to-text model failed"},
        502: {"description": "AI itinerary generation failed"},
    },
)
def plan_from_speech(
    audio: UploadFile = File(..., description="Spoken travel request, e.g. 'Plan three days in Paris with museums'"),
    current_user: User = Depends(get_current_user),
):
    """
    Transcribes a spoken travel request locally, then passes the text to the existing
    LangGraph itinerary workflow (RAG and weather tools, Anthropic model) and returns the
    structured itinerary with the transcript. The result is not saved; use `POST /trips`
    and `POST /itineraries` to create a persistent trip and itinerary.
    """
    transcript = _transcribe_upload(audio)
    if not transcript:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No speech was detected in the audio")
    try:
        itinerary = TravelPlanningOrchestrator().invoke(transcript)
    except Exception:
        logger.exception("Voice itinerary planning failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="AI generation failed")
    return VoicePlanResponse(transcript=transcript, **itinerary.model_dump())


def _transcribe_upload(audio: UploadFile) -> str:
    try:
        data = audio.file.read(transcription_service.max_bytes + 1)
        return transcription_service.transcribe(data, audio.content_type)
    except AudioTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc))
    except InvalidAudioError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except TranscriptionError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
