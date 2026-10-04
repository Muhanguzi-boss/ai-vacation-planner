import unittest

from app.services.itinerary_narration import narrate_itinerary

DAYS = [
    {
        "day": 1,
        "activities": [
            {
                "time": "morning",
                "activity": "Visit the Louvre and see the Mona Lisa.",
                "location": "Musée du Louvre, Rue de Rivoli, 75001 Paris",
                "estimated_cost": "$22",
            },
            {
                "time": "evening",
                "activity": "Dinner cruise on the Seine",
                "location": "Port de la Bourdonnais, Paris",
                "estimated_cost": "$80",
            },
        ],
    },
    {
        "day": 2,
        "activities": [
            {
                "time": "afternoon",
                "activity": "Walk around Montmartre",
                "location": "Montmartre, Paris",
                "estimated_cost": "$0",
            }
        ],
    },
]


class ItineraryNarrationTests(unittest.TestCase):
    def test_narration_is_deterministic_and_concise(self):
        narration = narrate_itinerary("Paris", DAYS, "$102", 3000)

        self.assertEqual(
            narration,
            "Here is your 2-day itinerary for Paris. "
            "Day 1. Morning: Visit the Louvre and see the Mona Lisa, at Musée du Louvre. "
            "Evening: Dinner cruise on the Seine, at Port de la Bourdonnais. "
            "Day 2. Afternoon: Walk around Montmartre. "
            "Estimated total cost: $102.",
        )

    def test_missing_cost_and_blank_activities_are_skipped(self):
        days = [{"day": 1, "activities": [{"time": "morning", "activity": "  ", "location": "Nowhere"}]}]

        self.assertEqual(narrate_itinerary("Rome", days, None, 3000), "Here is your 1-day itinerary for Rome. Day 1.")

    def test_long_itineraries_are_cut_at_day_boundaries(self):
        days = [
            {"day": n, "activities": [{"time": "morning", "activity": "Explore the city " * 10, "location": "Paris"}]}
            for n in range(1, 11)
        ]

        narration = narrate_itinerary("Paris", days, "$900", 600)

        self.assertLessEqual(len(narration), 600)
        self.assertRegex(narration, r"The remaining \d+ days are in your written itinerary\.")
        self.assertTrue(narration.endswith("Estimated total cost: $900."))

    def test_single_oversized_activity_is_hard_capped(self):
        days = [{"day": 1, "activities": [{"time": "morning", "activity": "word " * 2000, "location": "Paris"}]}]

        narration = narrate_itinerary("Paris", days, "$10", 500)

        self.assertLessEqual(len(narration), 500)


if __name__ == "__main__":
    unittest.main()
