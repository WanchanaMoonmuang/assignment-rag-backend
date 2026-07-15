# PRD: Knowledge Assistant

**Status:** Final — describes the implemented system. This is the single source of truth
for product behavior.

## 1. Purpose

Knowledge Assistant is a single-admin RAG demonstration application for ingesting business
documents and answering questions with Gemini. It supports asynchronous multi-format
ingestion, hybrid retrieval, cited answers with source inspection, and a calculator tool.
This document describes that implementation as one complete contract.

## 2. Objectives

The shipped system:

1. Ingests pasted text, `.txt`, `.pdf`, `.docx`, `.csv`, and `.json` asynchronously through
   a durable job queue, without blocking the API on conversion or embedding.
2. Extracts each format with a technique suited to that format's structure (not one
   universal pipeline — see §4.6), producing page/section/line/row/record-aware
   locations for citation.
3. Retrieves relevant chunks with hybrid lexical (BM25) and vector search, fused with
   MongoDB `$rankFusion`.
4. Lets the user control the global retrieved-chunk limit (Top K, `0-20`) for each prompt.
5. Uses foundation-model knowledge when appropriate while clearly attributing claims
   supported by uploaded documents.
6. Preserves citations and tool activity exactly when conversations are reopened.
7. Lets users inspect the exact extracted passage behind a citation, including the
   original PDF page.
8. Exposes safe latency and token metrics without logging user or document content.

## 3. Users and Product Experience

Target users: internal stakeholders, product and AI teams, developers, and business users
evaluating document-assisted question answering.

The product name displayed by the frontend is **Knowledge Assistant**. The interface is
light-mode only, responsive, and built with React 19 + MUI v9. On desktop, source
inspection uses a slide-over drawer so chat remains visible; on mobile it uses a
full-screen sheet with a back action.

## 4. Functional Requirements

### 4.1 Answer Generation

- Gemini is called through the Gemini Developer API or Vertex AI, selected by
  `GEMINI_PROVIDER`.
- Generation temperature is `GEMINI_TEMPERATURE`, default `0.2`, read from settings — never
  hardcoded in provider clients.
- The answer follows the language of the latest user message; document language does not
  override it.
- `HISTORY_CONTEXT_WINDOW` (`0-100`, default `8`) bounds how many prior messages are
  included; `0` disables history. History is loaded chronologically and excludes the
  message currently being processed.
- `build_prompt` keeps the final prompt within `GENERATION_CONTEXT_TOKEN_BUDGET` (default
  `32000` tokens) by trimming retrieved chunks first, then the oldest history messages.
  The latest user message is never removed.

#### Knowledge and attribution policy

- With retrieval disabled (`top_k=0`), Gemini answers from foundation knowledge and
  produces no document citations.
- With retrieval enabled, Gemini may blend foundation knowledge with retrieved evidence.
- Claims supported by retrieved evidence use inline numeric markers such as `[1]` (or
  `[2, 7]` for multiple sources) that map to returned sources. Unsupported foundation
  knowledge appears under a `General knowledge` heading without citation markers.
- The model must not imply that an unsupported foundation-knowledge claim came from an
  uploaded document.
- Empty or irrelevant retrieval results do not force a fixed fallback sentence; the model
  states the limitation and may still answer from general knowledge.

### 4.2 Top K

- The chat interface provides a Top K slider, range `0-20`, initialized from `GET
  /api/config` rather than a hardcoded frontend default.
- `0` skips query embedding and retrieval entirely.
- `1-20` is the maximum number of chunks returned by hybrid retrieval across all
  documents combined — Top K is not a document count, and users do not select documents.
- Retrieved chunks may come from one document or several; comparison answers use whatever
  documents the retrieved chunks represent.
- `RAG_TOP_K` (default `5`) is the backend default when a request omits `top_k`, validated
  at startup (`0-20`) — an invalid value prevents startup rather than being silently
  clamped.

### 4.3 Hybrid Retrieval

- Retrieval combines an Atlas `$vectorSearch` (index `MONGODB_VECTOR_INDEX`, path
  `embedding`, cosine similarity, dimensions = `GEMINI_EMBEDDING_DIMENSIONS`, default
  `768`) with an Atlas `$search` lexical/BM25 query (index `MONGODB_SEARCH_INDEX`),
  equal-weighted, fused with MongoDB `$rankFusion` (requires Atlas 8.0+).
- Duplicate chunk results are returned once.
- The fused result is sorted by descending combined relevance and limited to Top K.
- Each result includes the fused score and its format-aware location metadata.
- Index definitions are documented in `docs/mongodb-vector-index.json` and
  `docs/mongodb-search-index.json`.
- A missing or invalid index produces a diagnosable backend error and structured error
  log, not a misleading empty answer.

### 4.4 Document Ingestion

Supported inputs: pasted UTF-8 text, `.txt`, `.pdf`, `.docx`, `.csv`, `.json`.

File ingestion uses multipart upload. Conversion, chunking, embedding, and final
publication are performed by a separate worker process reading a MongoDB-backed job
queue (`ingestion_jobs`) — no Redis or other queue service is used.

#### Upload limit

- The maximum input size is exactly `20 MiB` (`MAX_UPLOAD_BYTES`).
- The frontend rejects an oversized file before sending; the backend remains
  authoritative and counts bytes in the file part while reading it. Client-supplied
  `Content-Length`/MIME values are advisory only.
- Pasted text is measured as UTF-8 bytes and uses the same limit.
- Rejection happens before conversion, extraction, GCS upload, job creation, chunking, or
  embedding; oversized requests return `413 Payload Too Large`.

#### Original files

- Original uploaded files are stored in a private GCS bucket, keyed deterministically by
  document ID.
- Originals are only accessible through authenticated backend endpoints — the frontend
  never receives a public bucket URL.
- Deleting a document removes its chunks and enqueues a `cleanup` job for the GCS object.
  The worker sweeps `cleanup_pending` jobs every 5 minutes, retrying up to 10 times; after
  10 failures the record is retained as `cleanup_failed` with an actionable log for
  operator cleanup. Searchable content is never restored once deleted.

### 4.5 Ingestion Jobs

States: `queued`, `processing`, `completed`, `failed`. While processing, a job reports one
of these stages where applicable: `converting`, `extracting`, `chunking`, `embedding`,
`finalizing`.

- Job claiming is atomic and uses a renewable lease (`INGESTION_JOB_LEASE_SECONDS`,
  default `300`, valid range `30-3600`) so an abandoned job can be retried after a worker
  failure.
- A job is attempted at most `INGESTION_JOB_MAX_ATTEMPTS` times (default `3`, range
  `1-10`).
- A processing attempt exceeding `INGESTION_PROCESSING_TIMEOUT_SECONDS` (default `600`,
  range `60-3600`) is terminated, recorded as a timeout, and retried within the attempt
  limit.
- Processing is idempotent: retries never publish duplicate documents or chunks. A
  document becomes searchable only after every required chunk and embedding is stored.
- Failed jobs retain a safe user-facing error code/message; internal stack traces are
  never returned.
- The frontend polls job state and shows the current stage, completion, or a clear
  failure message.

### 4.6 Extraction and Chunking

Extraction is **per-format, not a single universal pipeline.** `app/extraction.py:extract()`
dispatches on file extension:

| Format | Technique | Concept / why | Citation location |
| --- | --- | --- | --- |
| TXT | Line-window chunker, no conversion step | Plain text has no structure beyond lines | Line range |
| PDF | fitz/PyMuPDF directly (page-aware); table bounding boxes are detected first and excluded from the prose pass | Geometric exclusion prevents a table's cells being emitted both as prose and as a table chunk | Page (+ table index for table chunks) |
| DOCX | **The only format converted with MarkItDown**, then chunked by walking Markdown heading lines into a section breadcrumb | docx has no bespoke structure-parser in this codebase; converting to Markdown reuses the same heading/table-aware chunker | Section path |
| CSV | Row-batch chunker (≤20 rows/chunk) **plus one synthesized "Dataset summary" chunk** (row/column counts, column names, per-numeric-column min/max/mean) | Two chunk types: raw rows answer row-level questions, the summary chunk answers aggregate questions without the model scanning every row | Row range / "Dataset summary" |
| JSON | Record-aware chunker — one chunk per array element or one for a single root object; oversized records are split further | The retrieval unit matches the data's actual structure instead of a blind character window | Record index / JSON path |

Additional rules:

- Tables are retained as complete Markdown/pipe-table blocks when they fit the chunk
  limit; oversized tables split by rows, repeating headers in every resulting chunk.
- Invalid JSON and malformed CSV (missing/duplicate headers, mismatched row length)
  produce explicit ingestion failures.
- CSV decoding defaults to UTF-8 with UTF-8 BOM support.
- Encrypted or image-only scanned PDFs fail with an actionable message; a PDF that yields
  no extractable text also fails explicitly.
- User-supplied metadata is preserved on the document and propagated to chunk retrieval
  metadata.
- The backend inspects file content with format-specific parsers before conversion —
  browser-provided MIME types are advisory only, not trusted for dispatch.
- Every chunk records its type, sequence, source document, and format-aware location.

### 4.7 Citations and Source Inspection

Every persisted assistant message stores its complete source list. Reopening a
conversation renders the same citations without rerunning retrieval.

Each source contains: document ID and filename, chunk ID, snippet, retrieval score, chunk
type, and location (type, start/end where applicable, display label).

Clicking a citation opens the source drawer (`GET
/api/documents/{document_id}/chunks/{chunk_id}`), which returns the cited chunk plus
limited surrounding extracted content ("neighbors"). The drawer shows filename, location,
score, surrounding content, and an exact highlight of the cited chunk.

The same extracted-content viewer serves every format:

- PDF also has an Original tab — the frontend fetches the private file with its bearer
  token (`GET /api/documents/{document_id}/file`), creates a temporary blob URL, and opens
  it at `#page=N`; the blob URL is revoked when the viewer closes.
- DOCX shows the converted Markdown at the cited section.
- TXT/CSV/JSON show the cited line range / rows / record.

Highlighting uses escaped text segments and semantic `<mark>` elements; extracted content
is never inserted as unsanitized HTML.

### 4.8 Calculator Tool

- Gemini may invoke the calculator automatically for explicit arithmetic. Its function
  description and the system instruction both restrict it to concrete numeric
  computation — conceptual or document questions must not trigger it.
- Supported through both the Gemini Developer API and Vertex AI provider paths.
- The calculator accepts only a documented allowlist of numeric literals, operators,
  parentheses, and supported functions — no `eval`, no code execution, no network/file
  access.
- Streaming chat emits `tool_call`/`tool_result` events so the frontend shows progress
  before the answer completes; only resolved `tool_result` events are persisted with the
  assistant message (transient "requested" placeholders are streaming-only, so a
  reopened conversation never shows a stuck spinner).
- Invalid expressions return a controlled tool error Gemini may explain to the user.

### 4.9 Metrics and Trace Logs

The backend emits structured metrics containing: request/conversation identifiers,
provider and model name, retrieval/prompt-building/model-first-token/model-total/
request-total latency, token counts when reported, requested Top K and returned chunk
count, retrieval scores, tool names, and job stage durations/safe error identifiers.

Logs never contain prompts, chat message text, document text, extracted chunks,
snippets, calculator expressions/results, credentials, authorization headers, or raw
model tokens.

## 5. API Requirements

All endpoints except health and login require `Authorization: Bearer <token>`.

| Method & Path | Purpose |
| --- | --- |
| `GET /api/health` | Health check, no auth |
| `POST /api/auth/login` | Authenticate, return access token |
| `GET /api/auth/me` | Current user profile |
| `GET /api/config` | Safe runtime config: Top K default/range, upload limit, supported extensions |
| `POST /api/ingest` | **Deprecated**, kept for compatibility; enqueues a text ingestion job with the same validation as `/api/ingestions/text` |
| `POST /api/ingestions/text` | Start text ingestion, returns `202` + job ID |
| `POST /api/ingestions/file` | Start file ingestion (multipart), returns `202` + job ID; `415`/`413`/`422` for unsupported/oversized/malformed input |
| `GET /api/ingestions/{job_id}` | Job state, stage, safe progress/error info |
| `GET /api/documents` | List documents |
| `DELETE /api/documents/{document_id}` | Delete a document, its chunks, and enqueue GCS cleanup |
| `GET /api/documents/{document_id}/chunks/{chunk_id}` | Cited chunk + neighbors + location |
| `GET /api/documents/{document_id}/file` | Stream the private original |
| `GET /api/conversations` | List conversations |
| `GET /api/conversations/{conversation_id}` | Conversation history |
| `DELETE /api/conversations/{conversation_id}` | Delete a conversation |
| `POST /api/chat` | Non-streaming answer (debugging) |
| `POST /api/chat/stream` | SSE: `conversation`, `metadata`, `token`, `tool_call`, `tool_result`, `sources`, `done`, `error` |

`top_k` on chat requests is optional, validated `0-20`; omitting it uses `RAG_TOP_K`. The
completion event is sent only after the assistant message, sources, and tool traces are
persisted.

## 6. Data Requirements

### 6.1 Documents and Chunks

Documents carry ingestion status, source format, an original-object GCS reference, byte
size, and extraction metadata. Chunks carry chunk type, sequence, format-aware location,
and normalized content for source inspection. Any document that lacks source-location
metadata remains listable, searchable, and deletable, with basic filename/snippet
citations; rich location inspection requires re-ingestion.

### 6.2 Ingestion Jobs

Jobs store document/job identifiers, state, stage, attempts, lease owner/expiry,
timestamps, safe error details, and cleanup references. A unique job/document
relationship prevents duplicate publication.

### 6.3 Conversations

Assistant messages persist: final Markdown content, complete citation source objects,
calculator tool calls and results, and a creation timestamp. Conversation reads return
these fields unchanged so the frontend never depends on transient streaming state.

## 7. Frontend Requirements

- React 19, TypeScript, MUI v9 (Emotion), `react-markdown` + `rehype-sanitize` +
  `remark-gfm` for answers, TanStack Query for server state, `sessionStorage` for the
  demo bearer token.
- Read Top K defaults/limits and supported extensions from `GET /api/config`; send the
  selected Top K with every prompt.
- Validate file extension and byte size client-side before upload; treat backend
  validation as authoritative.
- Show ingestion jobs and their current stage without blocking the documents panel;
  self-clear transient success notices rather than showing them permanently.
- Keep confirmation dialogs for document and conversation deletion.
- Render persisted citations identically to freshly streamed ones, including multi-source
  markers like `[2, 7]` linkified to each cited source.
- Show source inspection in a slide-over drawer on desktop, a full-screen sheet on mobile.
- Never render extracted document content as raw HTML.
- Visualize multiple tool calls in one turn individually, both live and on reload.

## 8. Configuration

### 8.1 Backend (`app/settings.py`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_PROVIDER` | `developer_api` | `developer_api` or `vertex_ai` |
| `RAG_TOP_K` | `5` | Retrieval limit when a chat request omits Top K (`0-20`) |
| `HISTORY_CONTEXT_WINDOW` | `8` | Max prior messages in generation context (`0-100`) |
| `GEMINI_TEMPERATURE` | `0.2` | Generation temperature (`0.0-2.0`) |
| `GENERATION_CONTEXT_TOKEN_BUDGET` | `32000` | Prompt token ceiling before trimming |
| `MAX_UPLOAD_BYTES` | `20 MiB` | File/pasted-text ingestion limit |
| `GCS_BUCKET_NAME` | none | Private bucket for originals; required for file ingestion |
| `INGESTION_JOB_LEASE_SECONDS` | `300` | Worker recovery lease (`30-3600`) |
| `INGESTION_JOB_MAX_ATTEMPTS` | `3` | Bounded worker retries (`1-10`) |
| `INGESTION_PROCESSING_TIMEOUT_SECONDS` | `600` | Max duration of one conversion attempt (`60-3600`) |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `900` / `150` | Character chunk window/overlap for prose splitters |
| `MONGODB_VECTOR_INDEX` / `MONGODB_SEARCH_INDEX` | none / `document_chunks_text_index` | Atlas index names for hybrid retrieval |

Existing Gemini Developer API/Vertex AI, MongoDB, authentication, and CORS variables
remain as defined in the backend `.env.example`.

### 8.2 Frontend

`VITE_API_BASE_URL` defaults locally to `http://localhost:8080/api` and is baked in at
build time. Top K and upload constraints come from backend runtime configuration and are
not duplicated as frontend environment variables. No `.env` file, credential, private
bucket URL, MongoDB URI, Gemini key, or GCP service-account key may be committed.

## 9. Non-Goals

- User document selection or document-scoped chat.
- OCR for scanned PDFs or images; exact PDF-coordinate overlays; native Word rendering.
- Editing uploaded documents; Web URL ingestion.
- Additional user roles, registration, tenancy, or document permissions; conversation
  sharing.
- A general-purpose agent or arbitrary code execution.
- Redis or another external job queue.
- A full observability platform or RAG evaluation framework.
- Automatic re-ingestion of documents that predate source-location metadata.

## 10. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Worker interruption creates stuck or duplicate jobs | Atomic leases, bounded retries, idempotent finalization, partial-data cleanup |
| Conversion loses original layout | Highlight stored extracted chunks; locations are navigation aids, not exact overlays |
| Hybrid index is absent or misconfigured | Published Atlas index definitions; diagnosable errors instead of silent empty answers |
| Foundation knowledge mistaken for document evidence | Explicit prompt attribution policy; citations only from retrieved sources |
| PDF table cells duplicated between prose and table chunks | Table bounding boxes detected first and excluded from the prose pass (§4.6) |
| Top K and history increase latency or exceed context | Bounded chunks, default Top K 5, token-budget trimming, stage metrics |
| File conversion consumes excessive resources | Validate before enqueueing, process outside API workers, enforce worker timeouts |
| GCS and MongoDB operations diverge | Deterministic keys, tracked state, idempotency, cleanup retries |
| Older records lack rich source metadata | Preserve basic behavior; require re-ingestion for full inspection |

## 11. Acceptance Criteria

### 11.1 Ingestion

- TXT, PDF, DOCX, CSV, JSON, and pasted text complete through the durable worker and
  become searchable.
- A file or pasted payload above 20 MiB is rejected before conversion with a clear error.
- Worker restart recovery does not create duplicate documents or chunks.
- A conversion exceeding the processing timeout is terminated and retried within limits.
- Table fixtures retain headers and enough row context to answer table questions; no
  table's cell text also appears as a duplicate prose chunk.
- Citation locations match pages, sections, lines, rows, or records in extraction
  fixtures.
- Unsupported, malformed, encrypted, and image-only inputs fail clearly without
  searchable partial data.

### 11.2 Retrieval and Answers

- Top K zero performs no query embedding or retrieval and returns no citations.
- Top K `1-20` returns no more than the requested number of globally ranked chunks.
- An omitted Top K uses `RAG_TOP_K`; the frontend initializes from `/api/config`.
- Known lexical-only and semantic test queries return relevant hybrid results.
- The calculator does not trigger on conceptual/document questions, and does trigger on
  explicit arithmetic, including at `top_k=0`.
- Document-supported claims use inline markers mapped to persisted sources; mixed
  foundation knowledge appears under the uncited `General knowledge` heading.

### 11.3 Citations and Tools

- Citations remain visible and unchanged after leaving and reopening a conversation.
- Clicking a citation opens the correct location, score, and highlighted stored chunk.
- PDF citations open the original at the cited page.
- Malicious extracted markup renders as text and cannot execute.
- Calculator calls stream visible activity, reject unsafe expressions, and survive
  conversation reload without a stuck "running" state.
- Multi-source markers (`[2, 7]`) render as individually clickable citation links.

### 11.4 Operations and Compatibility

- Structured logs report stage latency, token counts when available, retrieval counts,
  scores, tool names, and safe errors without content leakage.
- Conversations and documents that predate source-location metadata remain readable and deletable.
- Backend tests and lint pass; frontend tests, typecheck, lint, and production build pass.

## 12. Tech Stack and Repository

### Backend — `assignment-rag-backend`

Python 3.12+, `uv`, FastAPI + Uvicorn (async), Pydantic/`pydantic-settings`, `pymupdf`
(fitz), `markitdown[docx]`, `langchain-text-splitters`, `motor` (MongoDB), `google-genai`,
`google-cloud-storage`, `pyjwt`. Dockerfile for container deployment (Cloud Run target).

### Frontend — `assignment-rag-frontend`

Node 24, React 19, TypeScript, Vite, MUI v9 + Emotion, `@tanstack/react-query`,
`react-markdown` + `rehype-sanitize` + `remark-gfm`, `eventsource-parser`. Dockerfile for
container deployment (Vercel target).

### Repository and branching

Two independently git-tracked repositories, each with `main` (stable/demo-ready,
protected) and `develop` (active development) branches. Feature branches are created from
`develop`; pull requests target `develop`, and `develop` merges into `main` for tagged
demo releases.

### Deployment

- Frontend: Vercel, `VITE_API_BASE_URL` supplied as a build-time argument.
- Backend: Google Cloud Run, running the API and worker in one container for this POC
  (a production deployment would run them as separate always-on services).
