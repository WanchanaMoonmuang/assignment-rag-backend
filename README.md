# Knowledge Assistant — Backend

FastAPI backend for **Knowledge Assistant**, a single-admin Retrieval-Augmented
Generation (RAG) chatbot: upload business documents, ask questions in a persistent
chat, and get Gemini-generated answers grounded in citations back to the source
material. Interactive API docs run at `/api/docs` once the server is running.

This backend implements the **V2** contract in
[`../docs/PRDv2.md`](../docs/PRDv2.md); see [`../docs/app-architecture.md`](../docs/app-architecture.md)
for the cross-repo architecture note, and
[`../docs/diagram-rag-data-flow.md`](../docs/diagram-rag-data-flow.md) for a visual
walkthrough of the pipeline described below.

## Contents

- [How the RAG pipeline works](#how-the-rag-pipeline-works)
- [Tech stack](#tech-stack)
- [Project layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Local setup](#local-setup)
- [Configuration](#configuration)
- [MongoDB Atlas search indexes](#mongodb-atlas-search-indexes)
- [API reference](#api-reference)
- [Checks](#checks)
- [Observability](#observability)
- [Deployment (Cloud Run POC)](#deployment-cloud-run-poc)

## How the RAG Pipeline Works

The backend is two cooperating processes that share one MongoDB database:

- **API** (`app.main`) — handles auth, chat, and accepts ingestion requests.
- **Worker** (`app.worker`) — a separate long-running process that does the actual
  document conversion, chunking, and embedding, claimed from a durable MongoDB job
  queue. Ingestion is asynchronous specifically so a large PDF never blocks an API
  request; the API returns a job ID immediately and the client polls it.

### Ingestion → indexing

1. `POST /api/ingestions/text` or `POST /api/ingestions/file` validates the input
   (size ≤ 20 MiB, extension/content-signature match for files), stores the
   original file in a private GCS bucket, and creates a `queued` job in MongoDB
   (`app/main.py`).
2. The worker claims the job with a renewable lease (so a crashed worker's job is
   automatically retried, up to `INGESTION_JOB_MAX_ATTEMPTS`) and walks it through
   `converting → extracting → chunking → embedding → finalizing`.
3. **Extraction is format-aware** (`app/extraction.py`): PDF is parsed page-by-page
   with PyMuPDF (page-number citations); DOCX is converted to Markdown via
   MarkItDown (section-path citations); TXT is split by line ranges; CSV is
   chunked in row batches with headers repeated per chunk; JSON is parsed into
   readable per-record chunks. Tables are kept as complete Markdown where they
   fit a chunk, or split by row with headers repeated.
4. **Chunking** for prose uses `langchain_text_splitters`, sized by
   `RAG_CHUNK_SIZE`/`RAG_CHUNK_OVERLAP` (defaults `900`/`150` characters).
5. Each chunk is embedded with Gemini's embedding model (`gemini-embedding-2`,
   `768` dimensions by default) and written to MongoDB alongside its text,
   location metadata, and source document reference. A document only becomes
   searchable once every chunk for it has been written successfully — a job
   never publishes a partially-embedded document.

### Query → answer

1. `POST /api/chat` / `POST /api/chat/stream` receives a question, an optional
   `top_k` (0–20; 0 skips retrieval entirely), and the conversation's recent
   history (last `HISTORY_CONTEXT_WINDOW` messages, default `8`).
2. **Retrieval is hybrid**: the question is embedded and run against MongoDB
   Atlas `$vectorSearch` (semantic similarity) in parallel with an Atlas `$search`
   full-text/BM25 query (lexical match), then the two ranked lists are combined
   with MongoDB's `$rankFusion` and truncated to the requested Top K
   (`app/main.py:retrieve_chunks`). This is deliberately global across all
   documents — Top K is a chunk budget, not a document filter, so a comparison
   question can pull evidence from several documents at once.
3. The prompt is built from chat history, the retrieved chunk text, and the
   question, trimmed to fit `GENERATION_CONTEXT_TOKEN_BUDGET` — retrieved chunks
   are dropped first, then the oldest history messages, before the latest
   question would ever be cut (`app/rag.py:build_prompt`).
4. Gemini (Vertex AI or the Developer API, selected by `GEMINI_PROVIDER`)
   generates the answer at `GEMINI_TEMPERATURE` (default `0.2`). It's instructed
   to mark document-supported claims with inline numeric citations like `[1]`
   matching the numbered sources in the prompt, and to place any unsupported
   general-knowledge claims under a `General knowledge` heading instead of
   fabricating a citation.
5. Gemini may also call a **restricted calculator tool** (`app/calculator.py`) —
   an allowlisted arithmetic expression evaluator, never `eval`, no code
   execution, no network/file access. Tool calls and results stream as SSE
   `tool_call`/`tool_result` events and are persisted with the message.
6. The response streams token-by-token over SSE, followed by the source list
   (document, chunk, snippet, score, and location) and a completion event. The
   assistant message — answer, sources, and tool activity — is persisted
   verbatim so reopening the conversation renders identical citations without
   re-running retrieval.
7. `GET /api/documents/{id}/chunks/{chunk_id}` lets a client fetch the exact
   stored chunk plus its immediate neighbors for source inspection, and
   `GET /api/documents/{id}/file` streams the private original (e.g. so the
   frontend can open a PDF at the cited page).

## Tech Stack

- **Framework**: FastAPI + Uvicorn, Python 3.12
- **Database**: MongoDB Atlas (documents, chunks/vectors, conversations, ingestion
  jobs) — also the vector store and lexical search index, via Atlas Vector Search
  + Atlas Search fused with `$rankFusion`
- **LLM**: Gemini, via `google-genai`, through either the Developer API or Vertex AI
- **Storage**: Google Cloud Storage for private original files
- **Extraction**: MarkItDown (DOCX and general conversion), PyMuPDF (page-aware
  PDF parsing), `langchain_text_splitters` (prose chunking)
- **Auth**: JWT bearer tokens (`pyjwt`), single demo admin account
- **Tooling**: `uv` (dependency management), `ruff` (lint), `pytest` (tests)

## Project Layout

```text
app/
  main.py            FastAPI routes: auth, chat, ingestion, documents, conversations
  rag.py              Gemini client, prompt building, SSE framing, calculator loop
  extraction.py       Format-specific document conversion and chunking
  worker.py           Ingestion job queue: claiming, leases, retries, cleanup
  calculator.py       Restricted arithmetic tool (no eval, no code execution)
  storage.py          GCS upload/download/delete for original files
  auth.py             Password check and JWT issuance/verification
  settings.py         All configuration, via pydantic-settings
  check_config.py     Startup sanity check: MongoDB reachable, indexes exist
  observability.py    Best-effort structured JSON event logging to stdout
  schemas.py          Pydantic request/response models
tests/
  test_app.py         API route and RAG-logic tests
  test_extraction.py  Per-format extraction/chunking tests
docs/
  mongodb-vector-index.json, mongodb-search-index.json   Atlas index definitions
  QA.md               Incremental QA log (test cases + bugs found)
  handoff.md           Point-in-time engineering handoff notes
```

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- A MongoDB Atlas cluster on **8.0+** with Vector Search and Search enabled
  (`$rankFusion` requires 8.0+)
- A Gemini Developer API key, **or** Google Cloud credentials for Vertex AI
- A Google Cloud Storage bucket for original file uploads

## Local Setup

```bash
uv sync --extra dev
cp .env.example .env        # then fill in the real values below
uv run uvicorn app.main:app --reload    # API on http://localhost:8080
uv run python -m app.worker              # ingestion worker, separate process
```

The API and worker are two independent processes — the API never spawns the
worker. Both must be running for file/text ingestion to complete; without the
worker, jobs stay `queued` forever.

Verify configuration and connectivity before relying on either process:

```bash
uv run python -m app.check_config
```

This pings MongoDB and confirms the configured Atlas Vector Search and Atlas
Search indexes exist on the chunks collection — see
[MongoDB Atlas search indexes](#mongodb-atlas-search-indexes) if they don't yet.

## Configuration

All settings are read from environment variables (`.env` locally; never commit
it) via `app/settings.py`. `AUTH_USERNAME=admin` / `AUTH_PASSWORD=adminRAG123`
matches the PRD's default demo account — set your own for anything beyond a demo.

**Required:**

| Variable | Notes |
| --- | --- |
| `GEMINI_PROVIDER` | `developer_api` (default) or `vertex_ai` |
| `GEMINI_API_KEY` | required when `GEMINI_PROVIDER=developer_api` |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | required when `GEMINI_PROVIDER=vertex_ai`; `GCP_PROJECT_ID` is accepted as a project fallback. Use `gcloud auth application-default login` locally, or the Cloud Run service account in deployment |
| `MONGODB_URI` | Atlas connection string |
| `MONGODB_DATABASE` | default `poc_rag` |
| `MONGODB_VECTOR_INDEX` | Atlas Vector Search index name (no default — must be set) |
| `MONGODB_SEARCH_INDEX` | Atlas Search index name, default `document_chunks_text_index` |
| `GCS_BUCKET_NAME` | private bucket for original files (file ingestion) |
| `JWT_SECRET_KEY` | at least 32 random characters |
| `AUTH_PASSWORD` | demo account password |
| `BACKEND_CORS_ORIGINS` | comma-separated allowed origins, e.g. `http://localhost:5173` |

**Tunable (sensible defaults, validated at startup — an invalid value prevents
startup rather than silently clamping):**

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_TOP_K` | `5` | chunk retrieval limit when a request omits `top_k` (`0`–`20`) |
| `HISTORY_CONTEXT_WINDOW` | `8` | prior messages included in generation context (`0`–`100`) |
| `GEMINI_TEMPERATURE` | `0.2` | generation temperature (`0.0`–`2.0`) |
| `GENERATION_CONTEXT_TOKEN_BUDGET` | `32000` | conservative token budget; chunks trimmed before history |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `900` / `150` | prose chunking window (characters) |
| `INGESTION_JOB_LEASE_SECONDS` | `300` | worker claim lease, `30`–`3600` |
| `INGESTION_JOB_MAX_ATTEMPTS` | `3` | bounded retries per job, `1`–`10` |
| `INGESTION_PROCESSING_TIMEOUT_SECONDS` | `600` | max single conversion attempt, `60`–`3600` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | JWT lifetime |

See `.env.example` for the complete list with placeholder values.

## MongoDB Atlas Search Indexes

Hybrid retrieval requires **MongoDB Atlas 8.0+** with two indexes on the chunk
collection (`MONGODB_CHUNK_COLLECTION`, default `document_chunks`):

1. **Vector Search index**, named by `MONGODB_VECTOR_INDEX` — create it from
   [`docs/mongodb-vector-index.json`](docs/mongodb-vector-index.json). Its
   dimensions must equal `GEMINI_EMBEDDING_DIMENSIONS` (`768` by default).
2. **Atlas Search index**, named by `MONGODB_SEARCH_INDEX` — create it from
   [`docs/mongodb-search-index.json`](docs/mongodb-search-index.json).

Retrieval fuses both with `$rankFusion` (equal-weighted) before applying Top K.
A missing or misconfigured index surfaces as a diagnosable `503` from
`/api/chat`, not a silent empty answer — `check_config` catches this before you
hit it live.

## API Reference

Full interactive docs (OpenAPI) are served at `/api/docs` when the API is
running. Summary:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | liveness check, no auth |
| `POST /api/auth/login` | exchange username/password for a bearer token |
| `GET /api/auth/me` | confirm the current token |
| `GET /api/config` | runtime Top K bounds, upload limit, supported extensions |
| `POST /api/ingestions/text` | start a text-ingestion job (`202`) |
| `POST /api/ingestions/file` | start a multipart file-ingestion job (`202`) |
| `GET /api/ingestions/{job_id}` | poll job status/stage/error |
| `POST /api/ingest` | **deprecated** V1 shim; delegates to the V2 text-ingestion job |
| `GET /api/documents` | list ingested documents |
| `DELETE /api/documents/{document_id}` | delete a document, its chunks, and its GCS original |
| `GET /api/documents/{document_id}/chunks/{chunk_id}` | fetch a cited chunk plus neighbors |
| `GET /api/documents/{document_id}/file` | stream the private original file |
| `GET /api/conversations` / `GET /api/conversations/{id}` / `DELETE /api/conversations/{id}` | conversation history |
| `POST /api/chat` | one-shot chat response |
| `POST /api/chat/stream` | SSE streaming chat (tokens, sources, tool activity) |

All endpoints except health and login require:

```text
Authorization: Bearer <access_token>
```

## Checks

```bash
uv run pytest              # unit/integration tests
uv run ruff check .        # lint
uv run python -m app.check_config   # MongoDB + Atlas index sanity check
```

## Observability

The API and worker write best-effort structured JSON events to stdout (Cloud
Run captures these without a separate telemetry service):

- `chat_metrics` — request/conversation IDs, provider/model, retrieval/prompt-
  build/first-token/model/request latency, input/output/total token counts
  (provider-reported when available, else estimated) and their source,
  effective Top K, retrieval count and scores, tool names, and a safe error
  code on failure.
- `worker_stage`, `worker_job`, `worker_cleanup` — each ingestion stage
  transition, attempt number, terminal status, safe error code, and duration.

These events deliberately **never** include chat text, prompts, answers,
document names/content/chunks/metadata, calculator expressions/results,
credentials, authorization headers, or raw provider tokens.

## Deployment (Cloud Run POC)

```bash
docker build -t knowledge-assistant-backend .
```

The image runs the API and worker in **one container** (`Dockerfile`'s `CMD`
starts both and exits if either dies). Configure the Cloud Run service with
always-allocated CPU and **min/max instances = 1** — otherwise the worker can
be stopped when there's no HTTP traffic, or multiple instances can needlessly
race to poll the same job queue. Set `BACKEND_CORS_ORIGINS` to the deployed
frontend's origin, and grant the service account access to Vertex AI (if used)
and the private GCS bucket.

This is a **POC topology**. Run the API and worker as separate, independently
scaled services for a production deployment.
