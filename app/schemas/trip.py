from pydantic import BaseModel
from datetime import datetime


class TripCreate(BaseModel):
    destination: str
    days: int
    budget: float | None = None
    trip_style: str | None = None


class TripUpdate(BaseModel):
    destination: str | None = None
    days: int | None = None
    budget: float | None = None
    trip_style: str | None = None


class TripResponse(BaseModel):
    id: int
    destination: str
    days: int
    budget: float | None
    trip_style: str | None
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True