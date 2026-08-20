# AI Vacation Planner API

Backend REST API built with FastAPI for planning vacations end-to-end.

## Tech Stack

- **FastAPI** — web framework
- **SQLAlchemy** — ORM
- **PostgreSQL** — database
- **JWT** — authentication
- **Pydantic** — data validation
- **SQLite vector store** — local semantic retrieval for travel knowledge

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

## Phase 4: RAG & Knowledge Systems

The itinerary pipeline uses retrieval-augmented generation (RAG):

```text
Documents -> chunks -> embeddings -> SQLite vector store
   -> semantic top-k search -> retrieved context -> LLM
   -> Pydantic validation -> saved itinerary
```

Seed Markdown documents live in `data/knowledge/`. Ingestion normalizes each document, splits it into overlapping chunks, and keeps destination, source, category, and chunk index metadata.

The default embedding provider is a deterministic local hashing embedder, so the project can be reviewed without paid APIs. Set `EMBEDDING_PROVIDER=openai` to use OpenAI embeddings with `OPENAI_API_KEY` and `EMBEDDING_MODEL`.

The vector store is a local SQLite database at `data/vector_store.db`. It stores chunk text, metadata, and embeddings, then ranks results using cosine similarity. PostgreSQL remains the application database.

### Ingest and test retrieval

From the repository root:

```bash
python -m scripts.ingest_knowledge
```

Ingestion is repeatable: the source filename and chunk index form an upsert key, so rerunning it updates chunks rather than duplicating them.

Test retrieval through Swagger or with:

```text
GET /knowledge/search?query=Paris%20museums%20and%20Metro&top_k=5
```

The response includes the query, chunk text, metadata, and cosine relevance score.

When `POST /itineraries` is called, the controller retrieves the trip, searches the knowledge base using destination and travel style, and passes ranked text to `app/services/ai_service.py`. The prompt separates trip parameters from retrieved context and treats documents as untrusted reference material, not executable instructions. Existing provider fallback, JSON parsing, Pydantic validation, and persistence remain active.

### RAG environment variables

| Variable                     | Default                  | Purpose                    |
| ---------------------------- | ------------------------ | -------------------------- |
| `KNOWLEDGE_BASE_PATH`        | `data/knowledge`         | Seed document directory    |
| `VECTOR_STORE_PATH`          | `data/vector_store.db`   | SQLite vector store path   |
| `EMBEDDING_PROVIDER`         | `local`                  | `local` or `openai`        |
| `EMBEDDING_MODEL`            | `text-embedding-3-small` | Remote embedding model     |
| `LOCAL_EMBEDDING_DIMENSIONS` | `2048`                   | Local vector size          |
| `KNOWLEDGE_CHUNK_SIZE`       | `900`                    | Maximum chunk characters   |
| `KNOWLEDGE_CHUNK_OVERLAP`    | `120`                    | Context overlap characters |
| `KNOWLEDGE_TOP_K`            | `5`                      | Number of results          |
| `KNOWLEDGE_MIN_SCORE`        | `0.2`                    | Minimum cosine score       |

## Architecture

app/
├── api/routes/ # Route handlers (auth, trips, itineraries)
├── core/ # Config, security (JWT, hashing)
├── db/ # Database engine and session
├── models/ # SQLAlchemy ORM models
├── schemas/ # Pydantic request/response schemas
├── services/ # AI, chunking, embeddings, knowledge, and vector store
├── data/knowledge/ # Seed travel documents
├── scripts/ # Repeatable ingestion commands
└── main.py # App entrypoint

## API Endpoints

| Method | Endpoint               | Description                      |
| ------ | ---------------------- | -------------------------------- |
| POST   | /auth/register         | Register a new user              |
| POST   | /auth/login            | Login and get JWT token          |
| GET    | /auth/me               | Get current user profile         |
| POST   | /trips                 | Create a trip                    |
| GET    | /trips                 | List all trips                   |
| GET    | /trips/{id}            | Get single trip                  |
| PUT    | /trips/{id}            | Update a trip                    |
| DELETE | /trips/{id}            | Delete a trip                    |
| POST   | /itineraries           | Create itinerary for a trip      |
| GET    | /itineraries/{trip_id} | Get itinerary for a trip         |
| GET    | /knowledge/search      | Semantic travel knowledge search |
