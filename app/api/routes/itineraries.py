from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.itinerary import ItineraryCreate, ItineraryResponse
from app.api.routes.auth import get_current_user
from app.models.user import User
from app.controllers.itinerary_controller import generate_and_save_itinerary, get_itinerary_by_trip

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