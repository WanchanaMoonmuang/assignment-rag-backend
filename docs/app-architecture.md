# Knowledge Assistant Architecture

## Purpose

Knowledge Assistant is a single-admin Retrieval-Augmented Generation (RAG) application. Users ingest business documents, ask questions in a persistent chat, and receive Gemini-generated answers with citations to the retrieved document chunks.

The system is split into two repositories:

| Repository | Responsibility |
| --- | --- |
| `assignment-rag-frontend` | React/MUI browser application for login, documents, conversations, streaming responses, and citations |
| `assignment-rag-backend` | FastAPI API for authentication, ingestion, MongoDB persistence and retrieval, Gemini calls, and SSE streaming |

[`PRD.md`](./PRD.md) is the product contract for the implemented system; this document describes the same implementation from an architectural angle.

## System Flow

```text
Browser (React + MUI)
  |  HTTPS/JSON and SSE, bearer token
  v
FastAPI API process ---- MongoDB Atlas: documents, chunks/vectors, conversations, ingestion_jobs
  |                          (hybrid retrieval: $vectorSearch + Atlas $search, fused with $rankFusion)
  |-- Gemini: query embeddings, answer generation, calculator tool loop
  `-- GCS: streams private originals for citation preview

Ingestion worker (separate process) ---- MongoDB: claims ingestion_jobs with a lease
  |-- fitz/PyMuPDF (.pdf), MarkItDown (.docx only), custom chunkers (.txt/.csv/.json)
  |-- Gemini: chunk embeddings
  `-- GCS: stores private originals
```

The API and the worker are two independent processes that coordinate only through
MongoDB — they never call each other directly. See `docs/diagram-system-architecture.md`
for the visual version with tech-stack labels, and `docs/diagram-rag-data-flow.md` for the
step-by-step ingestion/query flow.

## Install and Run

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js 24 LTS and npm
- A MongoDB Atlas cluster (8.0+, for `$rankFusion`) with a Vector Search index and an Atlas Search index
- A Gemini Developer API key, or Google Cloud credentials for Vertex AI
- A private Google Cloud Storage bucket for original files

### Backend

```bash
cd assignment-rag-backend
uv sync --extra dev
cp .env.example .env
uv run uvicorn app.main:app --reload --port 8080
uv run python -m app.worker   # separate process/terminal — the API does not spawn it
```

The API listens at `http://localhost:8080` by default. Interactive OpenAPI documentation is available at `http://localhost:8080/api/docs`.

Run backend checks with:

```bash
uv run pytest
uv run ruff check .
uv run python -m app.check_config
```

### Frontend

In a third terminal:

```bash
cd assignment-rag-frontend
nvm use
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:5173`. The frontend defaults to `http://localhost:8080/api` and stores the demo access token in `sessionStorage`; closing the browser session clears it.

Run frontend checks with:

```bash
npm run lint
npm run typecheck
npm test
npm run build
npm run test:e2e
```

## Configuration and Credentials

Create `assignment-rag-backend/.env` from `.env.example`. Never commit it.

### Gemini Developer API

```env
GEMINI_PROVIDER=developer_api
GEMINI_API_KEY=your_gemini_api_key
```

### Vertex AI

```env
GEMINI_PROVIDER=vertex_ai
GOOGLE_CLOUD_PROJECT=your_gcp_project
GOOGLE_CLOUD_LOCATION=us
```

For local Vertex AI usage, authenticate with Application Default Credentials, for example:

```bash
gcloud auth application-default login
```

Cloud Run should use its attached service account instead of a downloaded service-account key. `GCP_PROJECT_ID` is accepted as a fallback for `GOOGLE_CLOUD_PROJECT`.

### MongoDB, GCS, and Authentication

At minimum, configure:

```env
MONGODB_URI=your_atlas_connection_string
MONGODB_DATABASE=poc_rag
MONGODB_VECTOR_INDEX=document_chunks_vector_index
MONGODB_SEARCH_INDEX=document_chunks_text_index
GCS_BUCKET_NAME=your_private_bucket
AUTH_USERNAME=admin
AUTH_PASSWORD=your_demo_password
JWT_SECRET_KEY=a_random_value_at_least_32_characters_long
BACKEND_CORS_ORIGINS=http://localhost:5173
```

`MONGODB_DATABASE` (`poc_rag`) and the backend package name in `pyproject.toml`
(`poc-rag-backend`) are the actual current identifiers in the code — they predate the
`assignment-rag-*` repository rename and are left as-is rather than renamed in this doc,
since renaming them here without renaming the code would itself be inaccurate.

Create the Atlas Vector Search index on `document_chunks` from `assignment-rag-backend/docs/mongodb-vector-index.json`, and the Atlas Search index from `docs/mongodb-search-index.json`. The vector index dimensions must equal `GEMINI_EMBEDDING_DIMENSIONS` (`768` by default).

The frontend uses only:

```env
VITE_API_BASE_URL=http://localhost:8080/api
```

Do not put model keys, database credentials, passwords, or bearer tokens in frontend variables. `VITE_API_BASE_URL` is baked in at build time (a Docker `--build-arg`, not a runtime env var).

## Documents, Chunking, and Indexing

Ingestion is asynchronous. `POST /api/ingestions/text` or `POST /api/ingestions/file`
validates the request, creates a document record plus an `ingestion_jobs` record
(`queued`), and returns immediately. The separate worker process claims the job with a
renewable lease, converts/extracts/chunks/embeds it, and publishes the document only once
every chunk is written. Clients poll `GET /api/ingestions/{job_id}` for status. The
legacy `POST /api/ingest` endpoint remains available and enqueues the same kind of text
job, but is deprecated in favor of `/api/ingestions/text`.

- Maximum input size: exactly `20 MiB` (`MAX_UPLOAD_BYTES`), enforced by the backend
  while reading the file part — client-supplied `Content-Length`/MIME are advisory only.
- Original files are stored in a private GCS bucket; deleting a document enqueues a
  `cleanup` job that removes the GCS object (swept every 5 minutes, up to 10 retries).
- Supported formats: pasted text, `.txt`, `.pdf`, `.docx`, `.csv`, `.json`.

**Extraction is not one universal pipeline.** `app/extraction.py:extract()` dispatches on
file extension to a dedicated function per format, each with a different technique and a
different reason for that technique:

| Format | Function | Technique | Concept / why | Citation location |
| --- | --- | --- | --- | --- |
| `.txt` | `extract_text` | Line-window chunker, no conversion | Plain text has no structure beyond lines, so a sliding window over lines is sufficient | Line range |
| `.pdf` | `extract_pdf` | **fitz/PyMuPDF directly — never MarkItDown.** Detect table bounding boxes first (`page.find_tables()`), then extract prose from `page.get_text("blocks")` *excluding* any block overlapping a table bbox | Geometric exclusion is what stops a table's cells from being emitted twice — once as prose, once as a table chunk (the duplication bug this doc's own history caught) | Page (and table index for table chunks) |
| `.docx` | `extract_markdown` | **The only branch that uses MarkItDown.** Converts the file to Markdown first, then walks heading lines to build a section breadcrumb | docx has no bespoke structure-parser in this codebase; converting to Markdown reuses the same heading/table-aware chunker as any other Markdown source | Section path (e.g. `Refund Policy > Eligibility`) |
| `.csv` | `extract_csv` | Row-batch chunker (≤20 rows/chunk) **plus one synthesized "Dataset summary" chunk** (row/column counts, column names, per-numeric-column min/max/mean) | Two chunk types, not one: raw rows answer row-level questions, the summary chunk answers aggregate questions ("how many rows", "average X") without making the model scan every row chunk | Row range (rows), "Dataset summary" (summary) |
| `.json` | `extract_json` | Record-aware chunker: one chunk per array element, or one chunk for a single root object; oversized records are further split | The retrieval unit is the record, matching how JSON is actually structured, not a blind character window that could cut a record in half | Record index / JSON path |

Worked examples (illustrative, not literal chunk sizes):

```
PDF page 3, raw layout:                    Extracted chunks:
┌───────────────────────────┐
│ Intro paragraph...        │              ├─ prose chunk (Page 3)
│ ┌────┬────┬────┐          │   ──────►      "Intro paragraph... Closing paragraph."
│ │ Q1 │ Q2 │ Q3 │  table   │                (table's bbox excluded from this text)
│ │120 │150 │ 90 │          │
│ └────┴────┴────┘          │              └─ table chunk (Page 3, Table 1)
│ Closing paragraph...      │                 "Q1 | Q2 | Q3\n120 | 150 | 90"
└───────────────────────────┘
```

```
policy.docx → MarkItDown → Markdown:       Extracted chunks:
# Refund Policy
## Eligibility                             ├─ chunk (section: "Refund Policy > Eligibility")
Customers may request a refund   ──────►   │   "Customers may request a refund..."
within 30 days.
## Table of rates                          └─ chunk (section: "Refund Policy > Table of rates")
| Tier | Refund % |                            "Tier | Refund %\nA | 100"
| A    | 100      |
```

```
orders.csv:                                Extracted chunks:
customer,amount,region
Alice,120,US                               ├─ chunk 0 (Dataset summary)
Bob,90,EU                     ──────►      │   "Dataset: 3 rows, 3 columns.
Cara,150,US                                │    Columns: customer, amount, region.
                                            │    Numeric summary: amount: min=90, max=150, mean=120"
                                            │    (only numeric columns are summarized —
                                            │     "customer" and "region" are skipped)
                                            └─ chunk 1 (rows 1-3, table)
                                                "customer | amount | region\nAlice|120|US\nBob|90|EU\nCara|150|US"
```

```
[{"id":1,"name":"Alice"},                  Extracted chunks:
 {"id":2,"name":"Bob"}]        ──────►      ├─ chunk (record 1, $[0]) — {"id":1,"name":"Alice"}
                                             └─ chunk (record 2, $[1]) — {"id":2,"name":"Bob"}
```

```
notes.txt, 120 lines           ──────►      ├─ chunk (lines 1-40)
(RAG_CHUNK_SIZE / OVERLAP                   ├─ chunk (lines 33-73)   ← overlap with previous
 windows over lines)                        └─ chunk (lines 66-120)
```

Every chunk records its type, sequence, source document, and format-aware location.
Document metadata (`documents`) is stored separately from chunk content
(`document_chunks`); deleting a document transactionally deletes both plus enqueues GCS
cleanup.

## Retrieval

MongoDB Atlas is both the operational database and the search index. `retrieve_chunks`
(`app/main.py`) runs a `$rankFusion` combining an Atlas `$vectorSearch` (index
`MONGODB_VECTOR_INDEX`, path `embedding`, cosine similarity, dimensions =
`GEMINI_EMBEDDING_DIMENSIONS`, default `768`) with an Atlas `$search` lexical/BM25 query
(index `MONGODB_SEARCH_INDEX`), equal-weighted, then applies Top K. Duplicate chunks are
returned once; each result carries the fused score plus its format-aware location.

- Top K is a request-level value from `0` to `20` (`rag_top_k` in `settings.py`,
  default `5`, validated at startup — an invalid value fails startup rather than being
  silently clamped).
- `top_k=0` skips query embedding and retrieval entirely; the answer comes from Gemini's
  foundation knowledge with no citations.
- `top_k=1..20` limits the total number of globally ranked chunks — it is not a document
  filter, and results may span one or several documents.
- The frontend reads its slider default/range from `GET /api/config` rather than
  duplicating it.

## Answer Generation and Conversation History

Conversations are stored in MongoDB. Before processing a new question, the backend loads
the most recent `HISTORY_CONTEXT_WINDOW` messages (`0-100`, default `8`; `0` disables
history), appends the new user message, and builds a prompt with history, retrieved
context, and the question. `build_prompt` (`app/rag.py`) keeps the final prompt within
`GENERATION_CONTEXT_TOKEN_BUDGET` (default `32000` tokens) by trimming retrieved chunks
first, then the oldest history messages — the latest user message is never removed.

Gemini generates through either the Developer API or Vertex AI, at `GEMINI_TEMPERATURE`
(default `0.2`). `POST /api/chat` returns one completed JSON response for debugging;
`POST /api/chat/stream` streams SSE events (`conversation`, `metadata`, `token`,
`tool_call`, `tool_result`, `sources`, `done`).

Knowledge and attribution policy:

- With retrieval disabled (`top_k=0`), Gemini answers from foundation knowledge with no
  citations.
- With retrieval enabled, Gemini may blend retrieved evidence with foundation knowledge:
  document-supported claims use inline numeric markers like `[1]` mapped to the returned
  sources; unsupported general-knowledge claims appear under a `General knowledge`
  heading without markers.
- Gemini may call the calculator tool for explicit arithmetic. Its function-declaration
  description and the system instruction both restrict it to concrete numeric
  computation — conceptual or document questions should not trigger it. The tool
  accepts only an allowlisted expression grammar (no `eval`, no code execution, no
  network/file access). Tool calls and results are persisted with the assistant message
  so a reopened conversation shows the same resolved activity (only resolved
  `tool_result` events are persisted — transient `tool_call`/"requested" placeholders are
  streaming-only).

## Citations

Every persisted assistant message stores its complete source list (document ID/name,
chunk ID, snippet, score, chunk type, and location), so reopening a conversation renders
identical citations without rerunning retrieval.

Clicking a citation opens the source drawer (`GET
/api/documents/{document_id}/chunks/{chunk_id}`), which returns the cited chunk plus
limited surrounding extracted content ("neighbors"). The same viewer is used for every
format, with the cited passage highlighted via escaped text and a semantic `<mark>`
element — extracted content is never inserted as unsanitized HTML.

- PDF additionally offers an Original tab: the frontend fetches the private file with its
  bearer token (`GET /api/documents/{document_id}/file`), creates a temporary blob URL,
  and opens it at `#page=N`; the blob URL is revoked when the viewer closes.
- DOCX shows the converted Markdown at the cited section path.
- TXT/CSV/JSON show the cited line range / rows / record.

## API and Service Boundary

The browser is responsible for presentation, client-side validation, session-scoped
token storage, and rendering JSON/SSE responses. The backend owns authentication,
authorization, source-of-truth validation, persistence, chunking, embeddings, retrieval,
prompt construction, model calls, and citation creation.

All endpoints except health and login require `Authorization: Bearer <token>`. CORS is
explicitly configured through `BACKEND_CORS_ORIGINS`.

Current API surface (`app/main.py`):

- `GET /api/health` — no auth
- `POST /api/auth/login`, `GET /api/auth/me`
- `GET /api/config` — safe runtime config (Top K default/range, upload limit, supported extensions)
- `POST /api/ingest` — deprecated, kept for compatibility; enqueues a text ingestion job
- `POST /api/ingestions/text`, `POST /api/ingestions/file`, `GET /api/ingestions/{job_id}`
- `GET /api/documents`, `DELETE /api/documents/{document_id}`
- `GET /api/documents/{document_id}/chunks/{chunk_id}`, `GET /api/documents/{document_id}/file`
- `GET /api/conversations`, `GET /api/conversations/{conversation_id}`, `DELETE /api/conversations/{conversation_id}`
- `POST /api/chat`, `POST /api/chat/stream`

## Assumptions

- This is a single-admin demonstration, not a multi-user or multi-tenant product.
- MongoDB Atlas 8.0+ supports transactions, Vector Search, Atlas Search, and `$rankFusion`.
- Gemini embedding dimensions remain compatible with the configured Atlas index.
- Document content is trusted only after backend validation; browser MIME declarations are advisory.
- GCS originals are private; the frontend never receives a public object URL.
- The local frontend and backend run at ports `5173` and `8080` respectively unless overridden.
- The API and worker are two processes sharing one MongoDB deployment; the API never spawns the worker.

## Known Limitations

- No OCR for scanned PDFs or images; encrypted or image-only PDFs fail with an actionable error.
- No user-selected document filtering — Top K limits chunks globally, not per document.
- No arbitrary code execution: the calculator is restricted to an allowlisted expression grammar.
- No user registration, multi-user roles, tenancy, conversation sharing, or Web URL ingestion.
- No exact PDF-coordinate highlight overlays or native Word rendering — citation location is page/section/row/record-level, not pixel-level.
- Documents ingested before source-location metadata existed show basic filename/snippet citations only; rich inspection requires re-ingestion.
