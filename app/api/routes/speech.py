from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.routes.auth import get_current_user
from app.models.user import User
from app.schemas.speech import SpeechSynthesisRequest
from app.services.speech_service import InvalidSpeechTextError, SpeechSynthesisError, speech_service

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
