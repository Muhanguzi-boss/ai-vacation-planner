"""Deterministic spoken narration of a saved itinerary, for text-to-speech. Makes no LLM calls."""

from typing import Any


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).rstrip(" .")


def _describe_day(day: dict[str, Any]) -> str:
    sentences = [f"Day {day.get('day', '')}."]
    for activity in day.get("activities") or []:
        description = _clean(activity.get("activity"))
        if not description:
            continue
        place = _clean(str(activity.get("location") or "").split(",")[0])
        time_of_day = _clean(activity.get("time")).capitalize()
        sentence = f"{time_of_day}: {description}" if time_of_day else description
        if place and place.lower() not in description.lower():
            sentence += f", at {place}"
        sentences.append(sentence + ".")
    return " ".join(sentences)


def narrate_itinerary(destination: str, days: list[dict[str, Any]], total_estimated_cost: str | None, max_chars: int) -> str:
    days = days or []
    opening = f"Here is your {len(days)}-day itinerary for {_clean(destination)}."
    closing = f"Estimated total cost: {_clean(total_estimated_cost)}." if _clean(total_estimated_cost) else ""
    reserve = len(closing) + 80

    parts = [opening]
    for index, day in enumerate(days):
        description = _describe_day(day)
        if len(" ".join(parts)) + len(description) + reserve > max_chars:
            remaining = len(days) - index
            parts.append(
                f"The remaining {remaining} {'days are' if remaining != 1 else 'day is'} in your written itinerary."
            )
            break
        parts.append(description)
    if closing:
        parts.append(closing)

    narration = " ".join(parts)
    if len(narration) > max_chars:
        narration = narration[: max_chars - 1].rsplit(" ", 1)[0] + "."
    return narration
