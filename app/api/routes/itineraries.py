from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user
from app.controllers.itinerary_controller import generate_and_save_itinerary, get_itinerary_by_trip
from app.db.database import get_db
from app.models.user import User
from app.schemas.itinerary import ItineraryCreate, ItineraryDBResponse

router = APIRouter(prefix="/itineraries", tags=["Itineraries"])


@router.post("", response_model=ItineraryDBResponse, status_code=201)
def create_itinerary(
    data: ItineraryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Trigger AI itinerary generation for a trip and return the persisted result."""
    itinerary = generate_and_save_itinerary(db, data.trip_id, current_user.id)
    return ItineraryDBResponse(
        id=itinerary.id,
        trip_id=itinerary.trip_id,
        destination=itinerary.trip.destination,
        days=itinerary.days,
        total_estimated_cost=itinerary.total_estimated_cost or "N/A",
    )


@router.get("/{trip_id}", response_model=ItineraryDBResponse)
def get_itinerary(
    trip_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch the generated travel itinerary for a specific trip."""
    itinerary = get_itinerary_by_trip(db, trip_id, current_user.id)
    return ItineraryDBResponse(
        id=itinerary.id,
        trip_id=itinerary.trip_id,
        destination=itinerary.trip.destination,
        days=itinerary.days,
        total_estimated_cost=itinerary.total_estimated_cost or "N/A",
    )