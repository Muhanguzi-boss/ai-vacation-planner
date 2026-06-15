from pydantic import BaseModel
from typing import List


class Activity(BaseModel):
    time: str
    activity: str
    location: str
    estimated_cost: str


class ItineraryDay(BaseModel):
    day: int
    activities: List[Activity]


class ItineraryCreate(BaseModel):
    trip_id: int


class ItineraryResponse(BaseModel):
    id: int
    trip_id: int
    destination: str
    days: List[ItineraryDay]
    total_estimated_cost: str

    class Config:
        from_attributes = True