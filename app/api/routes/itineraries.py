from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.schemas.itinerary import ItineraryCreate, ItineraryResponse
from app.api.routes.auth import get_current_user
from app.models.user import User
from app.controllers.itinerary_controller import generate_and_save_itinerary, get_itinerary_by_trip
from app.services.itinerary_narration import narrate_itinerary
from app.services.speech_service import InvalidSpeechTextError, SpeechSynthesisError, speech_service

router = APIRouter(prefix="/itineraries", tags=["Itineraries"])


@router.post("", response_model=ItineraryResponse, status_code=201)
def create_itinerary(
    data: ItineraryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Triggers AI itinerary generation for a trip, saves it, and returns the structured itinerary.
    """
    itinerary = generate_and_save_itinerary(db, data.trip_id, current_user.id)
    return ItineraryResponse(
        id=itinerary.id,
        trip_id=itinerary.trip_id,
        destination=itinerary.trip.destination,
        days=itinerary.days,
        total_estimated_cost=itinerary.total_estimated_cost or "N/A"
    )


@router.get("/{trip_id}", response_model=ItineraryResponse)
def get_itinerary(
    trip_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fetches the generated travel itinerary for a specific trip.
    """
    itinerary = get_itinerary_by_trip(db, trip_id, current_user.id)
    return ItineraryResponse(
        id=itinerary.id,
        trip_id=itinerary.trip_id,
        destination=itinerary.trip.destination,
        days=itinerary.days,
        total_estimated_cost=itinerary.total_estimated_cost or "N/A"
    )


@router.post(
    "/{trip_id}/audio",
    response_class=Response,
    summary="Narrate an itinerary as audio",
    responses={
        200: {
            "description": "WAV audio narrating the saved itinerary",
            "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
        },
        401: {"description": "Missing or invalid bearer token"},
        404: {"description": "Trip or itinerary not found"},
        500: {"description": "The local text-to-speech engine failed"},
    },
)
def narrate_itinerary_audio(
    trip_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Reads the trip's saved itinerary aloud. The narration is built directly from the stored
    itinerary (no AI call) and spoken by the local text-to-speech engine.
    """
    itinerary = get_itinerary_by_trip(db, trip_id, current_user.id)
    narration = narrate_itinerary(
        itinerary.trip.destination,
        itinerary.days,
        itinerary.total_estimated_cost,
        settings.MAX_TTS_TEXT_CHARS,
    )
    try:
        audio = speech_service.synthesize(narration)
    except (SpeechSynthesisError, InvalidSpeechTextError):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Speech synthesis failed")
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="itinerary-{trip_id}.wav"'},
    )
