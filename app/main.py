from fastapi import FastAPI
from app.db.database import engine, Base
from app.api.routes import auth

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Vacation Planner",
    description="Backend API for planning vacations with AI",
    version="1.0.0",
)

app.include_router(auth.router)

@app.get("/")
def root():
    return {"message": "AI Vacation Planner API is running"}