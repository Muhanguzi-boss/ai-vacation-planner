from fastapi import HTTPException
from sqlalchemy.orm import Session
import logging

from app.models.trip import Trip
from app.models.itinerary import Itinerary
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)


def generate_and_save_itinerary(db: Session, trip_id: int, current_user_id: int) -> Itinerary:
    """
    Orchestrates AI itinerary generation:
    1. Fetches trip details and validates ownership.
    2. Validates that an itinerary does not already exist.
    3. Calls the AI Service to generate a structured itinerary.
    4. Saves the generated itinerary (days JSON & total_estimated_cost) into the database.
    5. Returns the itinerary.
    """
    # 1. Fetch trip and check permissions
    trip = db.query(Trip).filter(Trip.id == trip_id, Trip.user_id == current_user_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    # 2. Check if an itinerary already exists for this trip
    existing = db.query(Itinerary).filter(Itinerary.trip_id == trip_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Itinerary already exists for this trip")

    # 3. Generate itinerary via AI Service
    logger.info(f"Generating itinerary for trip {trip_id} (destination: {trip.destination})...")
    try:
        generated = ai_service.generate_itinerary(
            destination=trip.destination,
            days=trip.days,
            budget=trip.budget,
            trip_style=trip.trip_style
        )
    except Exception as e:
        logger.error(f"Failed to generate itinerary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"AI generation failed: {str(e)}"
        )

    # 4. Save to database
    try:
        itinerary = Itinerary(
            trip_id=trip_id,
            days=generated["days"],
            total_estimated_cost=generated["total_estimated_cost"]
        )
        db.add(itinerary)
        db.commit()
        db.refresh(itinerary)
        return itinerary
    except Exception as e:
        db.rollback()
        logger.critical(f"Database error saving itinerary for trip {trip_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to save the generated itinerary to the database."
        )


def get_itinerary_by_trip(db: Session, trip_id: int, current_user_id: int) -> Itinerary:
    """
    Fetches the itinerary for a specific trip, validating permissions.
    """
    trip = db.query(Trip).filter(Trip.id == trip_id, Trip.user_id == current_user_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    itinerary = db.query(Itinerary).filter(Itinerary.trip_id == trip_id).first()
    if not itinerary:
        raise HTTPException(status_code=404, detail="Itinerary not found")

    return itinerary
