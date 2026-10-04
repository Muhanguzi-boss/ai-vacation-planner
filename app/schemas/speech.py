from typing import List

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.schemas.itinerary import ItineraryDay


class SpeechSynthesisRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=settings.MAX_TTS_TEXT_CHARS,
        description="Text to convert to speech.",
        examples=["Welcome to your trip itinerary."],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value.strip()


class TranscriptionResponse(BaseModel):
    text: str = Field(description="Transcribed speech. Empty when no speech was detected.")


class VoicePlanResponse(BaseModel):
    transcript: str = Field(description="What the speech-to-text model heard.")
    destination: str
    days: List[ItineraryDay]
    total_estimated_cost: str
