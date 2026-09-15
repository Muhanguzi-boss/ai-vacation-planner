import logging
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.itinerary import Itinerary
from app.models.trip import Trip
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)


def generate_and_save_itinerary(db: Session, trip_id: int, current_user_id: int) -> Itinerary:
    """Generate a structured itinerary, validate it, and save it to the database."""
    trip = db.query(Trip).filter(Trip.id == trip_id, Trip.user_id == current_user_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    existing = db.query(Itinerary).filter(Itinerary.trip_id == trip_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Itinerary already exists for this trip")

    logger.info("Generating itinerary for trip %s (destination: %s)", trip_id, trip.destination)
    try:
        generated = ai_service.generate_itinerary(
            destination=trip.destination,
            days=trip.days,
            budget=trip.budget,
            trip_style=trip.trip_style,
            trip_id=trip_id,
        )
    except Exception as exc:
        logger.error("Failed to generate itinerary: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"AI generation failed: {exc}") from exc

    try:
        itinerary = Itinerary(
            trip_id=trip_id,
            days=generated.get("itinerary", []),
            total_estimated_cost=generated.get("total_estimated_cost", "N/A"),
        )
        db.add(itinerary)
        db.commit()
        db.refresh(itinerary)
        return itinerary
    except Exception as exc:
        db.rollback()
        logger.critical("Database error saving itinerary for trip %s: %s", trip_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save the generated itinerary to the database.") from exc


def get_itinerary_by_trip(db: Session, trip_id: int, current_user_id: int) -> Itinerary:
    """Fetch the itinerary for a specific trip, validating permissions."""
    trip = db.query(Trip).filter(Trip.id == trip_id, Trip.user_id == current_user_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    itinerary = db.query(Itinerary).filter(Itinerary.trip_id == trip_id).first()
    if not itinerary:
        raise HTTPException(status_code=404, detail="Itinerary not found")

    return itinerary
