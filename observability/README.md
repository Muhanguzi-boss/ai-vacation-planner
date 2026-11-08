# Datadog integration

The application records telemetry events in `app/core/telemetry.py`. By default
(`TELEMETRY_LOG_ENABLED=true`), every event is written as one JSON line to stderr (logger
`vacation_planner.telemetry`). Set `TELEMETRY_LOG_ENABLED=false` to turn the JSON lines off,
or `TELEMETRY_ENABLED=false` to disable all telemetry, including metrics. When
`DD_METRICS_ENABLED=true`, the same events are also sent to a Datadog Agent as DogStatsD
metrics over UDP. The sender uses only the Python standard library; no Datadog package is
installed.

> **Status:** the metric format and the dashboard definition are verified by offline tests
> only. Delivery to a real Datadog account, and live Anthropic token values, have not been
> verified in this repository.

## 1. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEMETRY_ENABLED` | `true` | Master switch for all telemetry (JSON logs and metrics) |
| `TELEMETRY_LOG_ENABLED` | `true` | Structured JSON telemetry lines on stderr |
| `DD_METRICS_ENABLED` | `false` | Send metrics to the Datadog Agent over DogStatsD |
| `DD_AGENT_HOST` | `localhost` | Agent host (DogStatsD) |
| `DD_DOGSTATSD_PORT` | `8125` | Agent DogStatsD UDP port |
| `DD_SERVICE` | `ai-vacation-planner` | `service` tag on every metric |
| `DD_ENV` | `local` | `env` tag on every metric |
| `DD_VERSION` | `1.0.0` | `version` tag on every metric |

The application does not need a Datadog API key: the **Agent** holds it and forwards the
metrics. UDP is fire-and-forget, so if no Agent is listening the metrics are dropped and
requests are unaffected.

## 2. Run a Datadog Agent

Install the Agent for your platform (<https://docs.datadoghq.com/agent/>) with your own
`DD_API_KEY` and `DD_SITE`. DogStatsD listens on UDP 8125 by default. With Docker:

```bash
docker run -d --name dd-agent \
  -e DD_API_KEY=<your-datadog-api-key> \
  -e DD_SITE=datadoghq.com \
  -e DD_DOGSTATSD_NON_LOCAL_TRAFFIC=true \
  -p 8125:8125/udp \
  gcr.io/datadoghq/agent:7
```

Then enable metrics in the application's `.env` and restart it:

```dotenv
DD_METRICS_ENABLED=true
DD_AGENT_HOST=localhost
DD_DOGSTATSD_PORT=8125
DD_ENV=local
```

Never commit a Datadog API key; it belongs only in the Agent's configuration.

### Check what is sent, without Datadog

With `DD_METRICS_ENABLED=true` and no Agent running, listen on the port yourself and make a
request to the API:

```bash
python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('127.0.0.1', 8125)); print(s.recv(65535).decode())"
```

## 3. Import the dashboard

`observability/datadog-dashboard.json` uses the Datadog dashboard JSON format.

- **UI:** Dashboards → New Dashboard → open the settings (cog) menu → **Import dashboard
  JSON** → paste the file contents. (Importing into an existing dashboard replaces its widgets.)
- **API:**

  ```bash
  curl -X POST "https://api.datadoghq.com/api/v1/dashboard" \
    -H "Content-Type: application/json" \
    -H "DD-API-KEY: ${DD_API_KEY}" \
    -H "DD-APPLICATION-KEY: ${DD_APP_KEY}" \
    -d @observability/datadog-dashboard.json
  ```

  Use the API host for your Datadog site (for example `api.datadoghq.eu`).

The dashboard has `env` and `service` template variables (default `service:ai-vacation-planner`).

**Percentiles:** `*.duration_ms` metrics are distributions. The p50/p95 queries return data
only after you enable percentiles for each `vacation_planner.*.duration_ms` metric
(Metrics → Summary → metric → *Advanced* → enable percentiles). The `avg` series work
without this.

## 4. Metrics

Each event `<event>` produces `vacation_planner.<event>.count` (one per event) and one metric
per measurement. Every metric also carries `service`, `env` and `version`.

| Metric | DogStatsD type | Tags |
| --- | --- | --- |
| `vacation_planner.http.request.count` | count | `method`, `route`, `status_code`, `status_class`, `outcome`, `error_category` |
| `vacation_planner.http.request.duration_ms` | distribution | same as above |
| `vacation_planner.ai.generation.count` | count | `workflow`, `model`, `outcome`, `error_category` |
| `vacation_planner.ai.generation.duration_ms` | distribution | same as above |
| `vacation_planner.ai.generation.tool_iterations` | count | same as above (successful itineraries) |
| `vacation_planner.llm.request.count` | count | `model`, `node`, `outcome`, `error_category` |
| `vacation_planner.llm.request.duration_ms` | distribution | same as above |
| `vacation_planner.llm.request.input_tokens` | count | same as above (only when the provider reports usage) |
| `vacation_planner.llm.request.output_tokens` | count | same as above |
| `vacation_planner.llm.request.total_tokens` | count | same as above |
| `vacation_planner.llm.model_init.count` | count | `model`, `outcome`, `error_category` |
| `vacation_planner.tool.execution.count` | count | `tool`, `outcome`, `error_category` |
| `vacation_planner.tool.execution.duration_ms` | distribution | same as above |
| `vacation_planner.mcp.weather.fallback.count` | count | `failure_type` |

Tag values:

- `route`: the route template, e.g. `/itineraries/:trip_id/audio`, or `unmatched`.
- `workflow`: `travel_planning`, `vision_analysis`.
- `node`: `plan`, `finalize` (LangGraph nodes), or `standalone` (vision).
- `tool`: `travel_knowledge_search`, `travel_weather`, or `unsupported` for a tool name the
  model requested that does not exist. The requested name itself is never sent.
- `outcome`: `success`, `failure`.
- `error_category`: `none` on success. HTTP: `auth_error`, `not_found`, `validation_error`,
  `payload_too_large`, `unsupported_media_type`, `rate_limited`, `client_error`,
  `upstream_error`, `server_error`, `unhandled_exception`. Model, tool and generation errors
  are derived from the exception type: `rate_limited`, `timeout`, `auth_error`,
  `connection_error`, `bad_request`, `provider_error`, `mcp_error`, `validation_error`,
  `model_init_error`, `missing_api_key`, `tool_error`, `tool_limit`, `unsupported_tool`,
  `invalid_output`, `unexpected_error`.
- `failure_type`: `timeout`, `failed`.

Count metrics submitted through DogStatsD are stored by Datadog as rates; the dashboard
uses `.as_count()` to show totals.

## 5. Privacy

Only events, tags and measurements listed in `EVENT_CATALOG` are sent. Tag values are
normalised to short, lowercase Datadog-compatible strings. Prompts, transcripts, request and
response bodies, tool inputs and outputs, raw paths, query strings, user identifiers,
credentials and exception messages are never recorded. Offline tests in
`tests/test_datadog_metrics.py` and `tests/test_telemetry.py` enforce this.
