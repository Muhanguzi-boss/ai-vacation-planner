import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.models import user  # noqa: F401
from app.controllers.itinerary_controller import generate_and_save_itinerary
from app.services.ai_service import GeneratedItinerary


def sample_trip():
    return SimpleNamespace(
        id=7,
        destination="Paris",
        days=3,
        budget=1200.0,
        trip_style="culture",
        user_id=11,
    )


def sample_itinerary():
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
                            "location": "Paris",
                            "estimated_cost": "$25",
                        }
                    ],
                }
            ],
            "total_estimated_cost": "$25",
        }
    )


def database_for_trip(trip):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.side_effect = [trip, None]
    saved = SimpleNamespace(
        id=19,
        trip_id=trip.id,
        days=sample_itinerary().model_dump()["days"],
        total_estimated_cost="$25",
    )
    db.refresh.side_effect = lambda itinerary: setattr(itinerary, "id", saved.id)
    return db


class ItineraryControllerIntegrationTests(unittest.TestCase):
    @patch("app.controllers.itinerary_controller.TravelPlanningOrchestrator")
    def test_generation_invokes_orchestrator_and_persists_structured_result(
        self, orchestrator_class
    ):
        trip = sample_trip()
        db = database_for_trip(trip)
        orchestrator_class.return_value.invoke.return_value = sample_itinerary()

        result = generate_and_save_itinerary(db, trip.id, trip.user_id)

        orchestrator_class.return_value.invoke.assert_called_once()
        request = orchestrator_class.return_value.invoke.call_args.args[0]
        self.assertIn("3-day trip to Paris", request)
        self.assertIn("culture", request)
        db.add.assert_called_once()
        db.commit.assert_called_once()
        self.assertEqual(result.days, sample_itinerary().model_dump()["days"])
        self.assertEqual(result.total_estimated_cost, "$25")

    @patch("app.controllers.itinerary_controller.TravelPlanningOrchestrator")
    def test_orchestration_failure_maps_to_existing_ai_error(self, orchestrator_class):
        trip = sample_trip()
        db = database_for_trip(trip)
        orchestrator_class.return_value.invoke.side_effect = RuntimeError("tool failure")

        with self.assertRaises(HTTPException) as context:
            generate_and_save_itinerary(db, trip.id, trip.user_id)

        self.assertEqual(context.exception.status_code, 502)
        self.assertIn("AI generation failed", context.exception.detail)
        db.add.assert_not_called()
        db.commit.assert_not_called()

    @patch("app.controllers.itinerary_controller.TravelPlanningOrchestrator")
    def test_orchestration_failure_detail_does_not_expose_internal_errors(self, orchestrator_class):
        trip = sample_trip()
        orchestrator_class.return_value.invoke.side_effect = RuntimeError(
            "Weather lookup failed: MCP server process could not be started"
        )

        with self.assertRaises(HTTPException) as context:
            generate_and_save_itinerary(database_for_trip(trip), trip.id, trip.user_id)

        self.assertEqual(context.exception.detail, "AI generation failed")


if __name__ == "__main__":
    unittest.main()