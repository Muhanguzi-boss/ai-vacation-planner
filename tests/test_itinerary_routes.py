import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import itineraries
from app.api.routes.auth import get_current_user
from app.db.database import Base, get_db
from app.models.itinerary import Itinerary
from app.models.trip import Trip
from app.models.user import User
from app.services.ai_service import GeneratedItinerary


def sample_itinerary() -> GeneratedItinerary:
    return GeneratedItinerary.model_validate(
        {
            "destination": "Paris",
            "days": [
                {
                    "day": 1,
                    "activities": [
                        {
                            "time": "morning",
                            "activity": "Visit the Louvre",
                            "location": "Louvre, Paris",
                            "estimated_cost": "$25",
                        }
                    ],
                }
            ],
            "total_estimated_cost": "$25",
        }
    )


class ItineraryRouteTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db = self.session_factory()
        self.user = User(
            email="traveler@example.com",
            username="traveler",
            hashed_password="not-used-by-this-test",
        )
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)
        self.app = FastAPI()
        self.app.include_router(itineraries.router)
        self.app.dependency_overrides[get_db] = lambda: self.db
        self.app.dependency_overrides[get_current_user] = lambda: self.user
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.db.close()
        self.engine.dispose()

    def create_trip(self, user_id=None):
        trip = Trip(
            destination="Paris",
            days=1,
            budget=500.0,
            trip_style="culture",
            user_id=user_id or self.user.id,
        )
        self.db.add(trip)
        self.db.commit()
        self.db.refresh(trip)
        return trip

    def test_create_itinerary_returns_and_persists_structured_response(self):
        trip = self.create_trip()

        with patch(
            "app.controllers.itinerary_controller.TravelPlanningOrchestrator"
        ) as orchestrator:
            orchestrator.return_value.invoke.return_value = sample_itinerary()

            response = self.client.post("/itineraries", json={"trip_id": trip.id})

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.json(),
            {
                "id": 1,
                "trip_id": trip.id,
                "destination": "Paris",
                "days": [
                    {
                        "day": 1,
                        "activities": [
                            {
                                "time": "morning",
                                "activity": "Visit the Louvre",
                                "location": "Louvre, Paris",
                                "estimated_cost": "$25",
                            }
                        ],
                    }
                ],
                "total_estimated_cost": "$25",
            },
        )
        orchestrator.return_value.invoke.assert_called_once()
        self.assertEqual(
            orchestrator.return_value.invoke.call_args.kwargs,
            {"expected_destination": "Paris", "expected_days": 1},
        )

        saved = self.db.query(Itinerary).filter(Itinerary.trip_id == trip.id).one()
        self.assertEqual(saved.days, response.json()["days"])
        self.assertEqual(saved.total_estimated_cost, "$25")

    def test_create_itinerary_requires_authentication(self):
        del self.app.dependency_overrides[get_current_user]

        with patch(
            "app.controllers.itinerary_controller.TravelPlanningOrchestrator"
        ) as orchestrator:
            response = self.client.post("/itineraries", json={"trip_id": 1})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Not authenticated"})
        orchestrator.assert_not_called()

    def test_create_itinerary_hides_trips_owned_by_another_user(self):
        other_user = User(
            email="other@example.com",
            username="other",
            hashed_password="not-used-by-this-test",
        )
        self.db.add(other_user)
        self.db.commit()
        self.db.refresh(other_user)
        trip = self.create_trip(user_id=other_user.id)

        with patch(
            "app.controllers.itinerary_controller.TravelPlanningOrchestrator"
        ) as orchestrator:
            response = self.client.post("/itineraries", json={"trip_id": trip.id})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Trip not found"})
        orchestrator.assert_not_called()

    def test_create_itinerary_rejects_duplicate_request(self):
        trip = self.create_trip()
        self.db.add(
            Itinerary(
                trip_id=trip.id,
                days=sample_itinerary().model_dump()["days"],
                total_estimated_cost="$25",
            )
        )
        self.db.commit()

        with patch(
            "app.controllers.itinerary_controller.TravelPlanningOrchestrator"
        ) as orchestrator:
            response = self.client.post("/itineraries", json={"trip_id": trip.id})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": "Itinerary already exists for this trip"},
        )
        orchestrator.assert_not_called()

    def test_create_itinerary_returns_generic_error_when_generation_fails(self):
        trip = self.create_trip()

        with patch(
            "app.controllers.itinerary_controller.TravelPlanningOrchestrator"
        ) as orchestrator:
            orchestrator.return_value.invoke.side_effect = RuntimeError(
                "provider failure with sensitive detail"
            )

            with self.assertLogs(
                "app.controllers.itinerary_controller", level="ERROR"
            ) as captured:
                response = self.client.post("/itineraries", json={"trip_id": trip.id})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "AI generation failed"})
        self.assertNotIn("sensitive detail", response.text)
        self.assertNotIn("sensitive detail", "\n".join(captured.output))
        self.assertNotIn("Paris", "\n".join(captured.output))
        self.assertEqual(
            self.db.query(Itinerary).filter(Itinerary.trip_id == trip.id).count(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
