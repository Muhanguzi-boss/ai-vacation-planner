from sqlachemy.orm import relationship
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    trips = relationship("Trip", back_populates="owner", cascade="all, delete-orphan")
    created_at = Column(DateTime(timezone=True), server_default=func.now())