# AI Vacation Planner API

Backend REST API built with FastAPI for planning vacations end-to-end: users manage trips, and an Anthropic-powered LangGraph agent generates structured day-by-day itineraries using a RAG travel knowledge base and a weather tool (optionally served over MCP). Phase 6 adds image understanding, local speech-to-text and text-to-speech, spoken itinerary requests, and spoken itineraries.

## Provider Architecture

- **LLM provider: Anthropic only.** Itinerary planning (LangGraph via `langchain-anthropic`) and image understanding use the Claude model in `ANTHROPIC_MODEL`. There is no fallback to another LLM provider.
- **Embeddings: local by default.** The RAG store uses a deterministic local hashing embedder. OpenAI is used **only** as an optional embedding provider (`EMBEDDING_PROVIDER=openai`), which is why `openai` remains in `requirements.txt`; it is never used for text generation.
- **Speech:** faster-whisper (speech-to-text) and pyttsx3/SAPI5 (text-to-speech) run locally; no speech API is called.
- **Weather:** a local stub, or Open-Meteo through the MCP weather server.

`app/services/ai_service.py` holds the shared Anthropic configuration. Its `generate_itinerary()` method is the original Phase 2 single-prompt generator (Anthropic, with a rule-based fallback); it is kept as that phase's baseline, and the API generates itineraries through the Phase 5 LangGraph orchestrator instead.

## Tech Stack

- **FastAPI** — web framework
- **SQLAlchemy** — ORM
- **PostgreSQL** — database
- **JWT** — authentication
- **Pydantic** — data validation
- **SQLite vector store** — local semantic retrieval for travel knowledge
- **Anthropic Claude** (via `anthropic` / `langchain-anthropic`) — itinerary planning and image understanding
- **LangChain + LangGraph** — tool-calling itinerary orchestration
- **Model Context Protocol** (`mcp`) — standardized access to the weather tool server
- **Open-Meteo** — free, keyless geocoding and forecast data used by the MCP weather server
- **faster-whisper** — local speech-to-text
- **pyttsx3** (Windows SAPI5) — local text-to-speech

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
   # Edit .env with your database credentials and ANTHROPIC_API_KEY
```

   Text-to-speech uses Windows SAPI5 voices. The first speech-to-text request downloads the
   Whisper model (about 145 MB for `base`) into the Hugging Face cache.

5. Run the server

```bash
   uvicorn app.main:app --reload
```

6. Open Swagger UI at `http://127.0.0.1:8000/docs`. Protected endpoints accept either the
   **OAuth2PasswordBearer** login form (username/password) or a pasted JWT under **bearerAuth**
   in the Authorize dialog.

7. Run the tests (no API keys, network access, Whisper model, or Windows voices are required)

```bash
   python -m pytest tests
```

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

**Failure handling:** tool exceptions are not swallowed — `ToolNode` is configured with `handle_tool_errors=False`, so a failing tool call surfaces as an exception that the orchestrator wraps in `OrchestrationError`, which the controller maps to an HTTP 502 (`"AI generation failed"`; details are logged server-side), matching the existing AI-failure error contract. Phase 6 adds a weather-specific fallback inside the MCP weather tool (see below).

**Required configuration:** the orchestrator reuses the existing Anthropic settings already read by `app/services/ai_service.py` — no new environment variables or duplicated config parsing were introduced:

| Variable             | Default                       | Purpose                               |
| -------------------- | ------------------------------ | -------------------------------------- |
| `ANTHROPIC_API_KEY`  | *(required)*                   | Anthropic credential used by the graph |
| `ANTHROPIC_MODEL`    | `claude-haiku-4-5` in `.env.example` | Model used for planning, finalizing, and (Phase 6) image analysis; must support vision |
| `LLM_TEMPERATURE`    | `0.3`                          | Sampling temperature                   |
| `LLM_MAX_TOKENS`     | `3000`                         | Max tokens per model call              |

LangSmith tracing was evaluated and is **not** part of this phase: the project requirements for Phase 5 do not call for it, so no tracing integration or related configuration was added.

## Phase 6: Multimodal AI & MCP

Phase 6 adds image and voice input/output and moves weather access behind the Model Context Protocol. Every new capability is a separate service with its own protected endpoint; the Phase 5 LangGraph workflow is unchanged apart from where the `travel_weather` tool gets its data.

### A. Multimodal AI

**Image understanding** — `POST /vision/analyze` (`app/services/vision_service.py`)

```text
Image upload -> type / size / magic-byte validation -> base64 image block + travel-analysis instruction
   -> Anthropic Claude (same ChatAnthropic configuration as the itinerary agent)
   -> with_structured_output(ImageTravelInsights) -> validated JSON
```

- Accepts JPEG, PNG, GIF, WebP up to `MAX_IMAGE_UPLOAD_MB`. The file's first bytes must match the declared content type.
- Returns `is_travel_related`, `likely_destination`, `confidence` (0–1), `landmarks`, `setting`, `suggested_trip_style`, `description`.
- The image is analyzed once and is **not** placed in the LangGraph message history (the graph replays its history on every model call). `likely_destination` / `suggested_trip_style` map directly onto `POST /trips` (`destination`, `trip_style`).

**Speech-to-text** — `POST /speech/transcribe` (`app/services/transcription_service.py`)

- Local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the CPU (`WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE`). No cloud speech service.
- Accepts WAV, MP3, M4A/MP4, WebM, OGG and FLAC. The container is detected from the file's first bytes and must match the declared content type (`application/octet-stream` is accepted when the content is recognised audio).
- Limits: `MAX_AUDIO_UPLOAD_MB` and `MAX_AUDIO_DURATION_SECONDS`. Over-long audio is rejected after decoding and language detection (first 30 seconds), before full transcription runs.
- The model is loaded on the first request (downloading it once into the Hugging Face cache) and reused. Loading and transcription share one lock, so requests are processed one at a time.

**Text-to-speech** — `POST /speech/synthesize` (`app/services/speech_service.py`)

- Local pyttsx3 using Windows SAPI5; returns `audio/wav`. Text is limited to `MAX_TTS_TEXT_CHARS`.
- Each request gets a fresh engine inside its own COM initialization; synthesis is serialized process-wide, and the temporary WAV file is always deleted.

### B. MCP weather

```text
User request
    ↓
LangGraph (TravelPlanningOrchestrator)
    ↓
travel_weather LangChain tool            (same name and arguments as Phase 5)
    ↓
MCPWeatherService  (app/tools/mcp_tools.py)
    ↓
MCPClient          (app/services/mcp_client.py, stdio transport)
    ↓
Weather MCP server (mcp_servers/weather_server.py, tool: get_forecast)
    ↓
Open-Meteo geocoding + forecast APIs
```

**Why MCP here:** the weather capability becomes a standalone tool server with a published, typed contract (`get_forecast(location)` with an input and output schema) instead of code inside the API process. Any MCP client can discover and call it, the server can be replaced or reused without touching the agent, and the LLM never sees MCP: it keeps calling the same `travel_weather` tool.

- **Server:** `python -m mcp_servers.weather_server` (stdio). Geocodes the place name, fetches a 7-day daily forecast, and returns `location`, `latitude`, `longitude`, `timezone`, `summary`, `daily[]`. It contains no LLM logic and needs no API key. HTTP errors, timeouts, unknown places and malformed responses become short MCP tool errors; raw Open-Meteo responses are never returned.
- **Client:** starts the server with the app's own interpreter, calls the tool, and shuts the process down before returning. It exposes a synchronous `call_tool()` (each call runs on a private event loop; a Proactor loop on Windows), so the synchronous FastAPI/LangGraph code did not have to become async. Failures become `MCPClientError`.
- **Feature flag:** `MCP_WEATHER_ENABLED=false` (default) keeps the Phase 5 local weather tool. With `true`, `travel_weather` uses MCP.
- **Fallback:** weather is supplementary, so with `MCP_WEATHER_FALLBACK_ENABLED=true` (default) an MCP failure is logged and the tool answers with the local weather result labelled `"source": "local fallback"`, and the itinerary is still generated. Only MCP client errors trigger the fallback; any other exception still fails the request. With the fallback disabled, an MCP failure fails itinerary generation with HTTP 502, as in Phase 5.

### C. Voice pipeline

```text
Audio -> faster-whisper -> text -> LangGraph itinerary workflow (RAG + weather/MCP, Claude)
      -> structured itinerary -> deterministic narration -> pyttsx3 -> WAV
```

- `POST /speech/plan` performs the first half in one call: it transcribes the audio and passes the transcript to the existing `TravelPlanningOrchestrator`, returning the transcript plus the structured itinerary. It does not create a trip or save anything (use `POST /trips` + `POST /itineraries` for that).
- `POST /itineraries/{trip_id}/audio` performs the second half for a saved itinerary: it builds a short narration directly from the stored itinerary (no LLM call) and returns it as WAV through the same TTS service.

### D. Vision pipeline

```text
Image -> Anthropic vision -> ImageTravelInsights -> (client) POST /trips -> POST /itineraries
```

### E. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` (`.env.example`) | Claude model for planning and image analysis |
| `MAX_IMAGE_UPLOAD_MB` | `5` | Image upload limit for `/vision/analyze` |
| `MCP_WEATHER_ENABLED` | `false` | Route `travel_weather` through the MCP weather server |
| `MCP_WEATHER_FALLBACK_ENABLED` | `true` | Use local weather when the MCP call fails |
| `MCP_TIMEOUT_SECONDS` | `15` | Deadline for one MCP call, including server start-up |
| `WHISPER_MODEL` | `base` | faster-whisper model size |
| `WHISPER_COMPUTE_TYPE` | `int8` | faster-whisper CPU compute type |
| `MAX_AUDIO_UPLOAD_MB` | `10` | Audio upload limit |
| `MAX_AUDIO_DURATION_SECONDS` | `300` | Longest audio accepted for transcription |
| `MAX_TTS_TEXT_CHARS` | `3000` | Longest text accepted for speech synthesis |

### F. Phase 6 endpoints

All require a bearer token.

| Method | Path | Request | Response | Purpose |
| --- | --- | --- | --- | --- |
| POST | `/vision/analyze` | multipart `image` | JSON `ImageTravelInsights` | Extract travel information from a photo |
| POST | `/speech/transcribe` | multipart `audio` | JSON `{ "text" }` | Local speech-to-text |
| POST | `/speech/synthesize` | JSON `{ "text" }` | `audio/wav` | Local text-to-speech |
| POST | `/speech/plan` | multipart `audio` | JSON transcript + itinerary | Spoken request → itinerary (not saved) |
| POST | `/itineraries/{trip_id}/audio` | — | `audio/wav` | Narrate a saved itinerary |

Errors use generic messages (`"AI image analysis failed"`, `"Transcription failed"`, `"Speech synthesis failed"`, `"AI generation failed"`); the underlying exception is logged on the server, and file paths, provider errors and stack traces are never returned. Validation errors describe the problem with the input (type, size, duration, emptiness).

### Known limitations

- Each MCP weather call starts a new server process (about 3–4 s, mostly Python start-up) before the Open-Meteo requests; process reuse is not implemented.
- The forecast covers the next 7 days from today, not the trip's dates (trips have no start date).
- Speech-to-text and text-to-speech each process one request at a time on the CPU; the first transcription after start-up also loads (and on first ever use, downloads) the Whisper model.
- Text-to-speech uses Windows SAPI5 and the default voice; other platforms need a different pyttsx3 driver (e.g. `espeak-ng` on Linux).
- `/speech/plan` returns an unsaved itinerary.

## Architecture

```text
FastAPI (app/main.py)
    ↓
Routes (app/api/routes) → Controllers (app/controllers)
    ↓
Services (app/services)
    ↓
LangGraph TravelPlanningOrchestrator ──→ Anthropic Claude
    ↓
LangChain tools (app/tools)
    ↓
RAG knowledge search   |   travel_weather → local weather  or  MCP client → MCP weather server → Open-Meteo

Separate Phase 6 services, called from their own routes:
    Vision          vision_service         → Anthropic Claude (structured output)
    Speech-to-text  transcription_service  → faster-whisper (local)
    Text-to-speech  speech_service         → pyttsx3 / SAPI5 (local)
```

```text
app/
├── api/routes/   # auth, trips, itineraries (+ audio), knowledge, vision, speech
├── controllers/  # Itinerary orchestration between routes, DB, and the AI agent
├── core/         # Config, security (JWT, hashing)
├── db/           # Database engine and session
├── models/       # SQLAlchemy ORM models
├── schemas/      # Pydantic request/response schemas
├── services/     # AI, RAG, LangGraph agent, vision, speech, transcription, narration, MCP client
├── tools/        # LangChain tools (RAG, weather) and the MCP weather adapter
└── main.py       # App entrypoint
mcp_servers/      # Standalone MCP weather server (Open-Meteo)
data/knowledge/   # Seed travel documents
scripts/          # Repeatable ingestion commands
tests/            # Unit and integration tests (fakes for LLM, MCP, Whisper, SAPI5)
```

## API Endpoints

| Method | Endpoint                      | Auth | Description                              |
| ------ | ----------------------------- | ---- | ---------------------------------------- |
| POST   | /auth/register                | No   | Register a new user                      |
| POST   | /auth/login                   | No   | Login and get JWT token                  |
| GET    | /auth/me                      | Yes  | Get current user profile                 |
| POST   | /trips                        | Yes  | Create a trip                            |
| GET    | /trips                        | Yes  | List all trips                           |
| GET    | /trips/{id}                   | Yes  | Get single trip                          |
| PUT    | /trips/{id}                   | Yes  | Update a trip                            |
| DELETE | /trips/{id}                   | Yes  | Delete a trip                            |
| POST   | /itineraries                  | Yes  | Create itinerary for a trip              |
| GET    | /itineraries/{trip_id}        | Yes  | Get itinerary for a trip                 |
| POST   | /itineraries/{trip_id}/audio  | Yes  | Narrate a saved itinerary as WAV         |
| GET    | /knowledge/search             | No   | Semantic travel knowledge search         |
| POST   | /vision/analyze               | Yes  | Image → structured travel insights       |
| POST   | /speech/transcribe            | Yes  | Audio → text                             |
| POST   | /speech/synthesize            | Yes  | Text → WAV                               |
| POST   | /speech/plan                  | Yes  | Spoken request → itinerary (not saved)   |
