from sqlalchemy import Column, Integer, ForeignKey, JSON, String
from sqlalchemy.orm import relationship
from app.db.database import Base


class Itinerary(Base):
    __tablename__ = "itineraries"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False)
    days = Column(JSON, nullable=False)
    total_estimated_cost = Column(String, nullable=True)

    trip = relationship("Trip", back_populates="itinerary")