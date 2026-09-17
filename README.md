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

As of Phase 5, `POST /itineraries` no longer calls the knowledge base directly from the controller. Retrieval now happens as a tool call inside the LangGraph orchestration described below — the graph decides when travel knowledge is relevant and fetches it itself. Retrieved chunks are still treated as untrusted reference material, not executable instructions, and Pydantic validation and persistence remain active.

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

## Phase 5: Frameworks & Orchestration

Itinerary generation is driven by a LangGraph agent instead of a single prompt call. The graph decides which tools are actually needed for a given request instead of always calling every tool.

```text
POST /itineraries
  -> auth, trip lookup, ownership check, duplicate check   (unchanged from Phase 1-4)
  -> TravelPlanningOrchestrator.invoke(request)             (app/services/agent_service.py)
       -> plan (Anthropic model with tools bound)
            -> tool calls?  -> tools node -> back to plan
            -> no tool calls -> finalize
       -> finalize (Anthropic model with structured output)
  -> GeneratedItinerary (existing Pydantic model)
  -> existing persistence + API response
```

**Why LangGraph:** the planner needs to loop between "call a tool" and "reason about the result" an unknown number of times before it has enough context to answer. LangGraph's `StateGraph` models that loop explicitly (nodes + conditional edges) instead of hand-rolling a while-loop around raw tool-call parsing.

**Planner/orchestrator:** `TravelPlanningOrchestrator` (`app/services/agent_service.py`) builds a graph with three nodes:

- `plan` — an Anthropic chat model (via `langchain-anthropic`, reusing `ai_service`'s API key, model name, temperature, and max tokens) with tools bound. It decides whether to answer directly or call a tool.
- `tools` — a LangGraph `ToolNode` that executes whichever tool(s) the planner requested and appends the raw results to the message history.
- `finalize` — the same Anthropic model with `with_structured_output(GeneratedItinerary)`, called once the planner has no more tool calls to make. It sees the full message history, including every tool result, and returns the existing `GeneratedItinerary` schema.

**Available tools** (`app/tools/travel_tools.py`), both thin LangChain `StructuredTool` wrappers around existing Phase 3/4 services — no new retrieval or weather logic was introduced:

- `travel_knowledge_search` — wraps the existing RAG `knowledge_service.search(...)`.
- `travel_weather` — wraps the existing `weather_service.get_weather(...)`.

**Tool selection:** the planner model chooses which, if any, tools to call based on the user's request — a simple "plan my weekend" request can finish without any tool calls, while a request that references weather or destination facts triggers the relevant tool(s). Both tools can be used in the same request.

**Result propagation:** tool outputs are returned as messages appended to the shared `messages` state and routed back into the `plan` node, so the planner (and ultimately `finalize`) always sees the exact tool output, not a paraphrase of it.

**Tool-loop safety:** `MAX_TOOL_ITERATIONS = 3` caps how many planner→tool round trips can happen; exceeding it raises `OrchestrationError` instead of looping indefinitely.

**Failure handling:** tool exceptions are not swallowed — `ToolNode` is configured with `handle_tool_errors=False`, so a failing tool call surfaces as an exception that the orchestrator wraps in `OrchestrationError`, which the controller maps to an HTTP 502, matching the existing AI-failure error contract.

**Required configuration:** the orchestrator reuses the existing Anthropic settings already read by `app/services/ai_service.py` — no new environment variables or duplicated config parsing were introduced:

| Variable             | Default                       | Purpose                               |
| -------------------- | ------------------------------ | -------------------------------------- |
| `ANTHROPIC_API_KEY`  | *(required)*                   | Anthropic credential used by the graph |
| `ANTHROPIC_MODEL`    | `claude-3-5-sonnet-20240620`   | Model used for planning and finalizing |
| `LLM_TEMPERATURE`    | `0.3`                          | Sampling temperature                   |
| `LLM_MAX_TOKENS`     | `3000`                         | Max tokens per model call              |

LangSmith tracing was evaluated and is **not** part of this phase: the project requirements for Phase 5 do not call for it, so no tracing integration or related configuration was added.

## Architecture

app/
├── api/routes/ # Route handlers (auth, trips, itineraries, knowledge)
├── controllers/ # Orchestration between routes, DB, and the AI agent
├── core/ # Config, security (JWT, hashing)
├── db/ # Database engine and session
├── models/ # SQLAlchemy ORM models
├── schemas/ # Pydantic request/response schemas
├── services/ # AI, chunking, embeddings, knowledge, vector store, LangGraph agent
├── tools/ # LangChain tool wrappers around existing RAG/weather services
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
