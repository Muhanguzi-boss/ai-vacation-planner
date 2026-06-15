import os
import json
import logging
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.models import user, trip, itinerary
from app.controllers.itinerary_controller import generate_and_save_itinerary, get_itinerary_by_trip
from app.schemas.itinerary import ItineraryResponse

# Set up logging to print to console
logging.basicConfig(level=logging.INFO)

def main():
    print("Starting integration test...")
    db = SessionLocal()
    try:
        # Clear existing itineraries for trip_id 1 to ensure test is idempotent
        from sqlalchemy import text
        db.execute(text("DELETE FROM itineraries WHERE trip_id = 1;"))
        db.commit()

        trip_id = 1
        user_id = 1
        
        print(f"Triggering itinerary generation for trip_id={trip_id}, user_id={user_id}...")
        itinerary = generate_and_save_itinerary(db, trip_id, user_id)
        
        print("\nSuccessfully generated and saved itinerary!")
        print(f"Itinerary ID: {itinerary.id}")
        print(f"Trip ID: {itinerary.trip_id}")
        print(f"Total Estimated Cost: {itinerary.total_estimated_cost}")
        print("Days generated:")
        print(json.dumps(itinerary.days, indent=2))
        
        # Verify schema mapping
        print("\nVerifying schema response mapping...")
        response = ItineraryResponse(
            id=itinerary.id,
            trip_id=itinerary.trip_id,
            destination=itinerary.trip.destination,
            days=itinerary.days,
            total_estimated_cost=itinerary.total_estimated_cost or "N/A"
        )
        print("Schema validation succeeded! Response payload matches requirements:")
        print(response.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
    finally:
        db.close()
        print("Database session closed.")

if __name__ == "__main__":
    main()
