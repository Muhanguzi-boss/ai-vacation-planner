from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.itinerary import Itinerary
from app.models.trip import Trip
from app.schemas.itinerary import ItineraryCreate, ItineraryResponse
from app.api.routes.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/itineraries", tags=["Itineraries"])


@router.post("", response_model=ItineraryResponse, status_code=201)
def create_itinerary(
    data: ItineraryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    trip = db.query(Trip).filter(Trip.id == data.trip_id, Trip.user_id == current_user.id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    existing = db.query(Itinerary).filter(Itinerary.trip_id == data.trip_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Itinerary already exists for this trip")

    itinerary = Itinerary(
        trip_id=data.trip_id,
        days=[day.model_dump() for day in data.days],
    )
    db.add(itinerary)
    db.commit()
    db.refresh(itinerary)
    return itinerary


@router.get("/{trip_id}", response_model=ItineraryResponse)
def get_itinerary(
    trip_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    trip = db.query(Trip).filter(Trip.id == trip_id, Trip.user_id == current_user.id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    itinerary = db.query(Itinerary).filter(Itinerary.trip_id == trip_id).first()
    if not itinerary:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    return itinerary