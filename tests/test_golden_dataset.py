import json
import unittest
from pathlib import Path

from app.services.ai_service import GeneratedItinerary


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evaluation"
    / "golden-itineraries-v1.json"
)


class GoldenItineraryDatasetTests(unittest.TestCase):
    def test_dataset_has_versioned_and_well_formed_reference_cases(self):
        dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

        self.assertRegex(dataset["dataset_version"], r"^\d+\.\d+\.\d+$")
        self.assertGreaterEqual(len(dataset["cases"]), 4)
        case_ids = [case["id"] for case in dataset["cases"]]
        self.assertEqual(len(case_ids), len(set(case_ids)))

        for case in dataset["cases"]:
            with self.subTest(case=case["id"]):
                self.assertGreaterEqual(case["days"], 1)
                self.assertIn("request", case)
                self.assertIn("travel_style", case)
                expected = GeneratedItinerary.model_validate(case["expected_output"])
                self.assertEqual(expected.destination, case["destination"])
                self.assertEqual(len(expected.days), case["days"])
                for day in expected.days:
                    times = {activity.time.casefold() for activity in day.activities}
                    self.assertTrue({"morning", "afternoon", "evening"} <= times)
                    self.assertGreaterEqual(len(day.activities), 3)

    def test_dataset_covers_multiple_destinations_durations_budgets_and_styles(self):
        cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["cases"]

        self.assertGreaterEqual(len({case["destination"] for case in cases}), 4)
        self.assertGreaterEqual(len({case["days"] for case in cases}), 4)
        self.assertTrue(any(case["budget"] is None for case in cases))
        self.assertTrue(any(case["budget"] is not None for case in cases))
        self.assertGreaterEqual(len({case["travel_style"] for case in cases}), 4)


if __name__ == "__main__":
    unittest.main()
