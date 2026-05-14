from pydantic import BaseModel
from typing import List


class ItineraryDay(BaseModel):
    day: int
    activities: List[str]


class ItineraryCreate(BaseModel):
    trip_id: int
    days: List[ItineraryDay]


class ItineraryResponse(BaseModel):
    id: int
    trip_id: int
    days: List[ItineraryDay]

    class Config:
        from_attributes = True