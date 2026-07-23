import json
import logging
import os
import time
from typing import Any, Dict, Optional

from pydantic import ValidationError

from app.schemas.itinerary import ActivityDay, ItineraryResponse
from app.services.weather_service import get_weather

# Lazily import OpenAI and Anthropic to ensure runtime safety
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

logger = logging.getLogger(__name__)


def get_itinerary_prompt(destination: str, days: int, budget: float | None, trip_style: str | None, weather_context: str, attempt: int = 0) -> str:
    """Build a strict prompt that requests only structured JSON output."""
    budget_str = f"${budget:.2f}" if budget is not None else "flexible"
    style_str = trip_style or "general sightseeing"
    strictness = "You must return JSON only. No prose, no markdown, no code fences, no extra fields." if attempt == 0 else "This is a retry. Return valid JSON only with every field present. Do not include explanations."

    return f"""You are a professional travel planner.
Create a realistic itinerary for {destination} that is affordable, geographically sensible, and evenly distributed across {days} days.

Trip parameters:
- Destination: {destination}
- Number of Days: {days}
- Budget: {budget_str}
- Travel Style: {style_str}
- Weather: {weather_context}

Requirements:
1. Every activity must be realistic and located within {destination} only.
2. Activities should be budget-aware and fit within the provided budget.
3. The itinerary should be evenly distributed across days with a mix of morning, afternoon, and evening activities.
4. Keep travel realistic and avoid long jumps between distant neighborhoods.
5. Keep the plan practical for a traveler rather than overly ambitious.

Return exactly one raw JSON object with this schema:
{{
  "trip_id": 123,
  "itinerary": [
    {{
      "day": 1,
      "activities": ["Visit the museum", "Walk in the old town"]
    }}
  ]
}}

Rules:
- Do not return text outside JSON.
- Do not wrap the response in markdown code fences.
- The top-level keys must be exactly "trip_id" and "itinerary".
- Each item in "itinerary" must have a numeric "day" and an "activities" array of strings.
- {strictness}
"""


def clean_json_response(text: str) -> str:
    """Strip markdown fences and surrounding whitespace from the LLM output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def generate_mock_fallback(destination: str, days: int, budget: float | None, trip_style: str | None, trip_id: int) -> Dict[str, Any]:
    """Generate a deterministic fallback payload when the LLM output cannot be parsed."""
    style_str = trip_style or "general"
    itinerary = []
    for day in range(1, days + 1):
        itinerary.append(
            {
                "day": day,
                "activities": [
                    f"Explore the main sights of {destination}",
                    f"Enjoy a local meal in {destination}",
                    f"Relax at a neighborhood spot in {destination}"
                ],
            }
        )

    total_budget = f"${days * 45:.0f}" if budget is None else f"${budget:.0f}"
    return {
        "trip_id": trip_id,
        "itinerary": itinerary,
        "total_estimated_cost": total_budget,
    }


class AIService:
    """Generate and validate structured itineraries from an LLM provider."""

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("anthropic_api_key")

        self.default_provider = (os.getenv("LLM_PROVIDER") or os.getenv("llm_provider") or "openai").lower()
        self.temperature = float(os.getenv("LLM_TEMPERATURE") or os.getenv("llm_temperature") or "0.3")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS") or os.getenv("llm_max_tokens") or "3000")
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES") or os.getenv("llm_max_retries") or "2")
        self.retry_backoff_seconds = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS") or os.getenv("llm_retry_backoff_seconds") or "0.5")

        self.openai_model = os.getenv("OPENAI_MODEL") or os.getenv("openai_model") or "gpt-4o-mini"
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL") or os.getenv("anthropic_model") or "claude-3-5-sonnet-20240620"

        self.openai_client = OpenAI(api_key=self.openai_key) if (OpenAI and self.openai_key) else None
        self.anthropic_client = Anthropic(api_key=self.anthropic_key) if (Anthropic and self.anthropic_key) else None

    def _generate_with_openai(self, prompt: str) -> str:
        if not self.openai_client:
            raise ValueError("OpenAI client is not initialized. Please set OPENAI_API_KEY.")

        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a precise JSON-only travel planner agent."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _generate_with_anthropic(self, prompt: str) -> str:
        if not self.anthropic_client:
            raise ValueError("Anthropic client is not initialized. Please set ANTHROPIC_API_KEY.")

        response = self.anthropic_client.messages.create(
            model=self.anthropic_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system="You are a precise JSON-only travel planner agent. Always return raw valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _validate_and_normalize(self, raw_payload: Any, trip_id: int) -> Dict[str, Any]:
        if not isinstance(raw_payload, dict):
            raise ValueError("LLM output is not a JSON object")

        if "trip_id" not in raw_payload or raw_payload.get("trip_id") is None:
            raise ValueError("Missing required field: trip_id")

        if "itinerary" not in raw_payload or not isinstance(raw_payload.get("itinerary"), list):
            raise ValueError("Missing required field: itinerary")

        normalized_days = []
        for item in raw_payload["itinerary"]:
            if not isinstance(item, dict):
                raise ValueError("Each itinerary item must be an object")
            if "day" not in item or "activities" not in item:
                raise ValueError("Each itinerary item must include day and activities")
            activities = item["activities"]
            if not isinstance(activities, list) or not all(isinstance(activity, str) for activity in activities):
                raise ValueError("Each itinerary item must contain a string activity list")

            normalized_days.append(ActivityDay(day=int(item["day"]), activities=activities).model_dump())

        validated = ItineraryResponse(trip_id=int(raw_payload["trip_id"]), itinerary=normalized_days)
        return validated.model_dump()

    def _build_weather_context(self, destination: str) -> str:
        try:
            weather = get_weather(destination)
            return weather.get("summary", "weather is moderate")
        except Exception as exc:
            logger.warning("Weather lookup failed: %s", exc)
            return "weather is moderate"

    def generate_itinerary(self, destination: str, days: int, budget: float | None, trip_style: str | None, trip_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate and validate an itinerary, retrying on invalid JSON or missing fields."""
        trip_id = trip_id if trip_id is not None else 0
        weather_context = self._build_weather_context(destination)
        prompt = get_itinerary_prompt(destination, days, budget, trip_style, weather_context)

        providers = ["openai", "anthropic"] if self.default_provider == "openai" else ["anthropic", "openai"]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            for provider in providers:
                if provider == "openai" and not self.openai_client:
                    continue
                if provider == "anthropic" and not self.anthropic_client:
                    continue

                logger.info("Generating itinerary for trip %s using %s (attempt %s)", trip_id, provider, attempt + 1)
                try:
                    if provider == "openai":
                        raw_response = self._generate_with_openai(get_itinerary_prompt(destination, days, budget, trip_style, weather_context, attempt=attempt))
                    else:
                        raw_response = self._generate_with_anthropic(get_itinerary_prompt(destination, days, budget, trip_style, weather_context, attempt=attempt))

                    logger.info("Raw LLM output for trip %s: %s", trip_id, raw_response)
                    cleaned = clean_json_response(raw_response or "")
                    logger.info("Cleaned LLM output for trip %s: %s", trip_id, cleaned)

                    parsed = json.loads(cleaned)
                    logger.info("Parsed LLM output for trip %s: %s", trip_id, parsed)

                    validated = self._validate_and_normalize(parsed, trip_id)
                    validated["total_estimated_cost"] = validated.get("total_estimated_cost", f"${days * 45:.0f}")
                    logger.info("Validated itinerary for trip %s: %s", trip_id, validated)
                    return validated
                except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                    last_error = exc
                    logger.warning("Attempt %s for trip %s failed with %s", attempt + 1, trip_id, exc)
                    if attempt < self.max_retries:
                        logger.info("Retrying itinerary generation for trip %s after %s seconds", trip_id, self.retry_backoff_seconds)
                        time.sleep(self.retry_backoff_seconds)
                    continue
                except Exception as exc:
                    last_error = exc
                    logger.error("Unexpected LLM generation error for trip %s: %s", trip_id, exc, exc_info=True)
                    continue

        logger.warning("All LLM attempts failed for trip %s. Using fallback response. Last error: %s", trip_id, last_error)
        return generate_mock_fallback(destination, days, budget, trip_style, trip_id)


ai_service = AIService()
