# AI Vacation Planner API

Backend REST API built with FastAPI for planning vacations end-to-end: users manage trips, and an Anthropic-powered LangGraph agent generates structured day-by-day itineraries using a RAG travel knowledge base and a weather tool (optionally served over MCP). Phase 6 adds image understanding, local speech-to-text and text-to-speech, spoken itinerary requests, and spoken itineraries. Phase 7 adds AI guardrails, a golden evaluation dataset with an optional DeepEval suite, broader offline tests, privacy-safe telemetry, and optional Datadog metrics with a dashboard.

## Provider Architecture

- **LLM provider: Anthropic only.** Itinerary planning (LangGraph via `langchain-anthropic`) and image understanding use the Claude model in `ANTHROPIC_MODEL`. There is no fallback to another LLM provider.
- **Embeddings: local only.** The RAG store uses a deterministic local hashing embedder computed in-process; no embedding API or key is used. `EMBEDDING_PROVIDER` accepts only `local`, and any other value is rejected.
- **Speech:** faster-whisper (speech-to-text) and pyttsx3/SAPI5 (text-to-speech) run locally; no speech API is called.
- **Weather:** a local stub, or Open-Meteo through the MCP weather server.
- **Observability:** structured JSON telemetry on stderr; optional Datadog metrics over DogStatsD (disabled by default). No Datadog library is required.

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

   Starting the app loads the project-root `.env` into the process environment automatically
   (`app/core/env.py`, called first in `app/main.py`), so the Anthropic settings
   (`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) are picked up
   without exporting them in the shell. Variables already set in the process environment take
   precedence over `.env`; values are never printed or logged.

   Text-to-speech uses Windows SAPI5 voices. The first speech-to-text request downloads the
   Whisper model (about 145 MB for `base`) into the Hugging Face cache.

5. Run the server

```bash
   uvicorn app.main:app --reload
```

6. Open Swagger UI at `http://127.0.0.1:8000/docs`. Protected endpoints accept either the
   **OAuth2PasswordBearer** login form (username/password) or a pasted JWT under **bearerAuth**
   in the Authorize dialog.

7. Run the offline tests (no API keys, network access, Whisper model, or Windows voices are required)

```bash
   python -m pytest tests
```

   pytest keeps Datadog metrics out of the test run even if your `.env` or shell enables them
   (see [Phase 7 → Testing](#c-testing)).

8. Optional, **paid**: run the model-based DeepEval evaluation. See
   [Phase 7 → Evaluation](#a-evaluation) before running it.

## Phase 4: RAG & Knowledge Systems

The itinerary pipeline uses retrieval-augmented generation (RAG):

```text
Documents -> chunks -> embeddings -> SQLite vector store
   -> semantic top-k search -> retrieved context -> LLM
   -> Pydantic validation -> saved itinerary
```

Seed Markdown documents live in `data/knowledge/`. Ingestion normalizes each document, splits it into overlapping chunks, and keeps destination, source, category, and chunk index metadata.

Embeddings are produced by a deterministic local hashing embedder, so the project runs without any paid embedding API. It is the only supported embedding provider.

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
| `EMBEDDING_PROVIDER`         | `local`                  | Must be `local`            |
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

## Phase 7: Evaluation, Security & Testing

Phase 7 makes the AI workflow measurable and safer without changing the provider architecture
(Anthropic for generation and judging, local embeddings):

- **Evaluation:** a versioned golden dataset, offline dataset checks, and an optional DeepEval suite.
- **Safety:** prompt-injection defences, restricted tool use, and deterministic validation of every generated itinerary.
- **Testing:** offline tests for the API routes, the LangGraph workflow, guardrails, telemetry and Datadog formatting.
- **Observability:** latency, model, token, tool and failure telemetry, with optional Datadog metrics and a dashboard.

### A. Evaluation

The versioned golden dataset is stored at
[`data/evaluation/golden-itineraries-v1.json`](data/evaluation/golden-itineraries-v1.json)
(version `1.0.0`, five cases: Tokyo, Paris, Lisbon, Banff and Rome, 2–5 days, fixed and
flexible budgets, five travel styles). Each case has a request, destination, duration, budget,
travel style and a reference itinerary. Offline tests (`tests/test_golden_dataset.py`) validate
its version, schema, destination/duration coverage, and expected daily schedules without
contacting an LLM.

The DeepEval suite (`tests/evaluation/deepeval_itinerary_eval.py`) is intentionally named so it is
**not collected** by `python -m pytest tests`. For each golden case it generates an itinerary
through the existing Anthropic-backed LangGraph flow, then:

1. runs deterministic checks first: Pydantic validation, destination and duration match, and at
   least three activities per day covering morning, afternoon and evening;
2. scores the output with an Anthropic judge (`ChatAnthropic`, temperature 0) on four GEval
   criteria, each with a minimum score of **0.70**: relevance, completeness, consistency, and
   output-structure adherence.

Install the optional evaluation dependency in the active environment:

```bash
pip install -r requirements-evaluation.txt
```

Then run only the evaluation suite (from the repository root):

```bash
python -m pytest tests/evaluation/deepeval_itinerary_eval.py -v
```

> **Warning — paid API calls.** The suite loads the project `.env` automatically. If
> `ANTHROPIC_API_KEY` is set there (or in your shell), the command immediately makes paid
> Anthropic calls: the itinerary generation for every golden case plus the judge calls for its
> four criteria. It is skipped only when no key is available or `deepeval` is not installed. Do
> not use it in routine offline testing or CI unless that cost and network access are intended.
> The model is `ANTHROPIC_MODEL` (default `claude-haiku-4-5`) for both generation and judging;
> no other provider key is used.

### B. Safety and guardrails

Implemented in `app/services/agent_service.py`, `app/services/ai_service.py` and the controllers:

- **Prompt injection:** trusted system instructions restrict the agent to travel planning and
  forbid revealing instructions or internal data. The user request is passed as untrusted data
  (`{"untrusted_user_travel_request": ...}`), and knowledge results and tool outputs are treated
  as untrusted reference data whose embedded instructions must be ignored. The vision prompt
  tells the model to ignore instructions written in images.
- **Restricted actions:** the planner can call only `travel_knowledge_search` and
  `travel_weather`. A request for any other tool is not executed and is recorded as
  `unsupported_tool`. Planner→tool loops are capped at `MAX_TOOL_ITERATIONS = 3`.
- **Rejecting invalid output:** every itinerary is validated by `GeneratedItinerary` (non-blank
  fields, at least one day and one activity per day, days numbered consecutively from 1). For
  `POST /itineraries`, the destination (case- and whitespace-insensitive) and the number of days
  must also match the trip. Invalid output is rejected with HTTP 502 `"AI generation failed"`
  and nothing is saved.
- **Safe errors and logs:** clients receive generic error messages. Telemetry records only the
  exception type. The itinerary-generation, itinerary-save and image-analysis error logs do not
  include exception messages, and the Anthropic model-creation error never carries the
  provider's message (which could echo the API key).

### C. Testing

Run the offline suite from the repository root:

```bash
python -m pytest tests
```

It needs no API keys, Whisper model, Windows voices or PostgreSQL connection. `DATABASE_URL` and
`SECRET_KEY` must still be set (for example in `.env` from setup step 4) because the settings are
loaded at import time; no database connection is made. LLMs, MCP, Whisper and SAPI5 are replaced
with fakes, and the itinerary route tests use in-memory SQLite. It covers the
itinerary, audio, vision and speech routes; the LangGraph workflow (tool calls, injection
handling, tool limits, invalid output, destination/duration mismatches); the golden dataset;
local embeddings; `.env` loading; and telemetry and Datadog formatting, privacy and failure
isolation. Apart from loopback UDP in the Datadog tests (a datagram to a socket the test binds
itself on `127.0.0.1`), no network access is used.

**Datadog isolation:** `tests/conftest.py` sets `DD_METRICS_ENABLED=false` in the test process
before the application is imported. Process environment variables take precedence over `.env`,
so a developer's `DD_METRICS_ENABLED=true` cannot make the test run send metrics to a local
Agent or a Datadog account. An automatic check also fails any test that leaves a Datadog sink on
the shared telemetry object. The Datadog tests build their own sinks with fake or loopback
sockets. This isolation applies when tests run through pytest.

### D. Observability

`app/core/telemetry.py` records privacy-safe events:

| Event | What it measures |
| --- | --- |
| `http.request` | Every API request: latency, method, route template, status, error category |
| `ai.generation` | Itinerary planning and image analysis: latency, model, outcome, error category, tool iterations |
| `llm.request` | Each Anthropic call: latency, model, LangGraph node, input/output/total tokens when reported |
| `llm.model_init` | Anthropic model creation, including a missing API key |
| `tool.execution` | Each tool call: latency, tool name, outcome, error category |
| `mcp.weather.fallback` | MCP weather failures answered by the local fallback |

- **Structured logs:** by default (`TELEMETRY_LOG_ENABLED=true`), each event is written as one
  JSON line to stderr (logger `vacation_planner.telemetry`).
- **Datadog metrics (optional):** with `DD_METRICS_ENABLED=true`, the same events are sent to a
  Datadog Agent over DogStatsD (UDP) using only the Python standard library, as
  `vacation_planner.<event>.count` plus `vacation_planner.<event>.<measurement>` (latencies are
  distributions, tokens are counts). The application needs no Datadog API key; the Agent holds it.
  Sending never affects requests: failures are ignored.
- **Privacy:** only catalogued tags and measurements are sent; routes are templates such as
  `/itineraries/:trip_id`. Prompts, transcripts, raw paths, query strings, user data, exception
  messages and keys are never recorded.
- **Dashboard:** [`observability/datadog-dashboard.json`](observability/datadog-dashboard.json)
  covers HTTP traffic and failures, AI generation, Anthropic latency and token usage, model
  initialisation failures, tools and MCP fallbacks.

See [`observability/README.md`](observability/README.md) for running a Datadog Agent, the full
metric and tag list, importing the dashboard, and enabling percentiles. The integration is
verified by offline tests only; delivery to a real Datadog account has not been verified in this
repository.

### E. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEMETRY_ENABLED` | `true` | Master switch for all telemetry (JSON logs and metrics) |
| `TELEMETRY_LOG_ENABLED` | `true` | JSON telemetry lines on stderr |
| `DD_METRICS_ENABLED` | `false` | Send metrics to a Datadog Agent over DogStatsD (**disabled by default**) |
| `DD_AGENT_HOST` | `localhost` | Agent host (IPv4 is preferred when it resolves to both) |
| `DD_DOGSTATSD_PORT` | `8125` | Agent DogStatsD UDP port |
| `DD_SERVICE` | `ai-vacation-planner` | `service` tag on every metric |
| `DD_ENV` | `local` | `env` tag on every metric |
| `DD_VERSION` | `1.0.0` | `version` tag on every metric |

### F. Known limitations

- The morning/afternoon/evening schedule rule is checked by the evaluation suite, not by the
  production validator.
- Destination and duration matching is applied to `POST /itineraries`; `/speech/plan` relies on
  schema validation only.
- Auth, trips, `GET /itineraries/{trip_id}` and knowledge-search routes have no dedicated route tests.
- The DeepEval suite has not been run as part of this repository's offline verification, and the
  same model generates and judges itineraries.

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

Phase 7 telemetry (app/core/telemetry.py, app/services/llm_telemetry.py):
    request middleware + LLM/tool callbacks + generation events
        → JSON lines on stderr (default)
        → DogStatsD over UDP → Datadog Agent → Datadog (optional, DD_METRICS_ENABLED=true)
```

```text
app/
├── api/routes/   # auth, trips, itineraries (+ audio), knowledge, vision, speech
├── controllers/  # Itinerary orchestration between routes, DB, and the AI agent
├── core/         # Config, .env loading, security (JWT, hashing), telemetry and Datadog sink
├── db/           # Database engine and session
├── models/       # SQLAlchemy ORM models
├── schemas/      # Pydantic request/response schemas
├── services/     # AI, RAG, LangGraph agent, LLM/tool telemetry callbacks, vision, speech, transcription, narration, MCP client
├── tools/        # LangChain tools (RAG, weather) and the MCP weather adapter
└── main.py       # App entrypoint (loads .env first)
mcp_servers/      # Standalone MCP weather server (Open-Meteo)
data/knowledge/   # Seed travel documents
data/evaluation/  # Versioned golden itinerary dataset
observability/    # Datadog dashboard JSON and setup guide
scripts/          # Repeatable ingestion commands
tests/            # Offline unit and integration tests (fakes for LLM, MCP, Whisper, SAPI5)
tests/evaluation/ # Optional, paid DeepEval suite (not collected by `pytest tests`)
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
