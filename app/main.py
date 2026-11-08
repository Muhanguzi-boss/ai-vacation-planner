# Load .env into os.environ before any app module is imported: ai_service reads the
# Anthropic configuration from the environment at import time.
from app.core.env import load_local_env

load_local_env()

from fastapi import FastAPI  # noqa: E402
from fastapi.openapi.utils import get_openapi  # noqa: E402

from app.core.telemetry import RequestTelemetryMiddleware  # noqa: E402
from app.db.database import engine, Base  # noqa: E402
from app.models import user, trip, itinerary  # noqa: E402
from app.api.routes import auth, trips, itineraries, knowledge, vision, speech  # noqa: E402

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Vacation Planner",
    description="Backend API for planning vacations with AI",
    version="1.0.0",
)

app.add_middleware(RequestTelemetryMiddleware)

app.include_router(auth.router)
app.include_router(trips.router)
app.include_router(itineraries.router)
app.include_router(knowledge.router)
app.include_router(vision.router)
app.include_router(speech.router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="AI Vacation Planner",
        version="1.0.0",
        description="Backend API for planning vacations with AI",
        routes=app.routes,
    )
    openapi_schema["components"].setdefault("securitySchemes", {})["bearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }
    for path_item in openapi_schema["paths"].values():
        for operation in path_item.values():
            if {"OAuth2PasswordBearer": []} in operation.get("security", []):
                operation["security"].append({"bearerAuth": []})
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
def root():
    return {"message": "AI Vacation Planner API is running"}