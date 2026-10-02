from typing import List

from pydantic import BaseModel, Field


class ImageTravelInsights(BaseModel):
    is_travel_related: bool = Field(
        description="Whether the image shows a place, landmark, or scene useful for travel planning."
    )
    likely_destination: str | None = Field(
        default=None,
        description="Most likely city or region shown, e.g. 'Paris, France'. Null if it cannot be identified.",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in likely_destination, from 0 (no idea) to 1 (certain).",
    )
    landmarks: List[str] = Field(
        default_factory=list,
        description="Named landmarks or attractions visible in the image.",
    )
    setting: str = Field(
        description="Short description of the environment, e.g. 'urban', 'beach', 'mountains', 'countryside'."
    )
    suggested_trip_style: str | None = Field(
        default=None,
        description="Travel style the image suggests, e.g. 'culture', 'relaxation', 'adventure', 'food'.",
    )
    description: str = Field(description="One or two sentences describing the image for a travel planner.")
