from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.db.database import engine, Base
from app.models import user, trip, itinerary
from app.api.routes import auth, trips, itineraries, knowledge

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Vacation Planner",
    description="Backend API for planning vacations with AI",
    version="1.0.0",
)

app.include_router(auth.router)
app.include_router(trips.router)
app.include_router(itineraries.router)
app.include_router(knowledge.router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="AI Vacation Planner",
        version="1.0.0",
        description="Backend API for planning vacations with AI",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
def root():
    return {"message": "AI Vacation Planner API is running"}