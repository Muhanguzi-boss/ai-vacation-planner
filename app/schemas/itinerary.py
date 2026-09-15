from typing import List

from pydantic import BaseModel, ConfigDict


class ActivityDay(BaseModel):
    day: int
    activities: List[str]


class ItineraryResponse(BaseModel):
    trip_id: int
    itinerary: List[ActivityDay]


class ItineraryCreate(BaseModel):
    trip_id: int


class ItineraryDBResponse(BaseModel):
    id: int
    trip_id: int
    destination: str
    days: List[ActivityDay]
    total_estimated_cost: str

    model_config = ConfigDict(from_attributes=True)