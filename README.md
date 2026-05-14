# AI Vacation Planner API

Backend REST API built with FastAPI for planning vacations end-to-end.

## Tech Stack

- **FastAPI** — web framework
- **SQLAlchemy** — ORM
- **PostgreSQL** — database
- **JWT** — authentication
- **Pydantic** — data validation

## Setup

1. Clone the repo

```bash
   git clone https://github.com/YOUR_USERNAME/ai-vacation-planner.git
   cd ai-vacation-planner
```

2. Create and activate a virtual environment

```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
   pip install -r requirements.txt
```

4. Set up environment variables

```bash
   cp .env.example .env
   # Edit .env with your database credentials
```

5. Run the server

```bash
   uvicorn app.main:app --reload
```

6. Open Swagger UI at `http://127.0.0.1:8000/docs`

## Architecture

app/
├── api/routes/ # Route handlers (auth, trips, itineraries)
├── core/ # Config, security (JWT, hashing)
├── db/ # Database engine and session
├── models/ # SQLAlchemy ORM models
├── schemas/ # Pydantic request/response schemas
└── main.py # App entrypoint

## API Endpoints

| Method | Endpoint               | Description                 |
| ------ | ---------------------- | --------------------------- |
| POST   | /auth/register         | Register a new user         |
| POST   | /auth/login            | Login and get JWT token     |
| GET    | /auth/me               | Get current user profile    |
| POST   | /trips                 | Create a trip               |
| GET    | /trips                 | List all trips              |
| GET    | /trips/{id}            | Get single trip             |
| PUT    | /trips/{id}            | Update a trip               |
| DELETE | /trips/{id}            | Delete a trip               |
| POST   | /itineraries           | Create itinerary for a trip |
| GET    | /itineraries/{trip_id} | Get itinerary for a trip    |
