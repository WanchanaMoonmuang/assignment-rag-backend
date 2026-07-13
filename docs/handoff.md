# Backend Handoff

Last updated: 2026-07-13

## Current State

The backend implements the approved V2 scope for asynchronous ingestion, multi-format extraction, hybrid retrieval, configurable generation, citations, calculator tools, and structured observability. Unit and integration-style tests pass, but the latest real local curl run found two unresolved runtime failures. Do not treat the remaining feature stack as locally validated until those failures are fixed and the curl matrix is rerun.

Repository: `poc-rag-backend`

Current branch: `feature/backend-v2-observability`

Current committed head:

```text
c8e3b30 Add backend observability metrics
```

The only local modification at handoff is `docs/test_document.md`, containing the latest curl results. Preserve it.

## Source Documents

- Shared V2 product contract: `../docs/PRDv2.md`
- Shared architecture: `../docs/app-architecture.md`
- Shared metadata strategy: `../docs/metadata.md`
- Backend setup and API summary: `README.md`
- Curl scenarios and historical results: `docs/test_document.md`
- Search index definitions: `docs/mongodb-vector-index.json` and `docs/mongodb-search-index.json`

## Branch and Merge State

`origin/develop` already contains generation, durable ingestion, and the POC API/worker container. GitHub PR #3 was named `feature/backend-v2-extraction`, but its merged second parent is `b7ffb0c` (the container commit), not the later extraction commit `79b1012`. The actual extraction implementation remains unmerged. The remaining branches must be reviewed and merged in this order because each is based on the previous branch:

1. `feature/backend-v2-extraction` at `79b1012`
2. `feature/backend-v2-hybrid-retrieval` at `fa922e5`
3. `feature/backend-v2-citations-tools` at `d2b7ebe`
4. `feature/backend-v2-observability` at `c8e3b30`

Relevant history:

```text
c8e3b30 Add backend observability metrics
d2b7ebe Add citations and calculator tools
fa922e5 Add hybrid retrieval and runtime config
79b1012 Add file extraction worker pipeline
b7ffb0c Run API and worker in POC container
1531539 Add durable document ingestion jobs
55f96ff Add configurable chat generation
```

Before continuing, fetch remote state and verify it again:

```bash
git fetch origin develop
git log --oneline --decorate --graph --all -20
git branch -r --no-merged origin/develop
```

## Completed Scope

- FastAPI bearer authentication and conversation persistence.
- Chat history context controlled by `HISTORY_CONTEXT_WINDOW`, default `8`.
- Gemini Developer API and Vertex AI provider selection.
- Request Top K range `0-20`, defaulted from `RAG_TOP_K=5`; Top K `0` skips retrieval.
- MongoDB-backed durable ingestion jobs with leases, retries, timeouts, and terminal cleanup.
- API and worker can run in one POC container. This requires always-allocated CPU, minimum instances `1`, and maximum instances `1` on Cloud Run.
- Backend-authoritative 20 MiB limit for pasted text and file uploads.
- TXT, PDF, DOCX, CSV, and JSON extraction with type/location metadata.
- Private original-file storage through GCS. The project bucket is `poc-rag-58`.
- MongoDB Atlas hybrid vector and lexical retrieval using `$rankFusion`.
- Persisted citations, source-neighbor lookup, and authenticated original-file access.
- Calculator tool support in non-streaming and streaming generation.
- Structured JSON chat and worker metrics on stdout, including latency and provider token usage when available; safe estimates are the fallback.
- Logging call sites exclude prompts, messages, answers, document content, snippets, metadata, calculator values, credentials, and raw model tokens.

Scope 6 received repeated backend QA review. The final review was clear. The committed verification result was:

```text
ruff check .: passed
pytest: 60 passed, 1 Starlette/httpx deprecation warning
```

## Main Modules

- `app/main.py`: FastAPI routes, auth-protected API orchestration, chat, SSE, ingestion submission, documents, conversations, and citations.
- `app/rag.py`: Gemini clients, prompts, token usage, streaming, calculator orchestration, and RAG helpers.
- `app/worker.py`: MongoDB job claiming, leases, extraction/embedding/publication stages, failure handling, and GCS cleanup sweep.
- `app/extraction.py`: format-specific extraction and chunk metadata.
- `app/storage.py`: GCS upload, download, and deletion.
- `app/calculator.py`: restricted arithmetic evaluator; does not use `eval`.
- `app/observability.py`: best-effort JSON event emission to stdout.
- `app/settings.py`: validated environment settings.
- `app/check_config.py`: MongoDB and Atlas index validation; currently has a Vertex configuration bug described below.
- `tests/test_app.py` and `tests/test_extraction.py`: backend coverage.

## Required Local Configuration

Use `.env` for local runtime values but never commit it. Important settings are:

```text
GEMINI_PROVIDER=vertex_ai
GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID
GOOGLE_CLOUD_LOCATION=us
MONGODB_URI
MONGODB_DATABASE
MONGODB_VECTOR_INDEX
MONGODB_SEARCH_INDEX
GCS_BUCKET_NAME=poc-rag-58
JWT_SECRET_KEY
AUTH_USERNAME=admin
AUTH_PASSWORD
```

Application Default Credentials were working at handoff. Both direct Vertex calls below succeeded using model access from the existing environment:

- `GeminiClient.generate(...)`
- `GeminiClient.generate_with_calculator(...)`

The calculator call returned provider usage metadata: 102 input, 9 output, and 111 total tokens for the diagnostic prompt. Do not write access tokens, credentials, prompts, or document content into logs or committed documentation.

## Current Runtime Blockers

### 1. Chat API returns HTTP 500

On the feature stack, this request returned `500 Internal Server Error` in about 374 ms:

```http
POST /api/chat
{"question":"Reply with the word ready.","top_k":0}
```

Top K `0` bypasses retrieval, and direct Vertex generation plus direct calculator-enabled generation both succeeded. The failure is therefore in API orchestration, persistence, runtime process state, or another dependency around generation rather than basic Vertex access.

Run a clean feature-branch API in the foreground and capture its traceback while repeating the curl request. At handoff, port `8080` was occupied by an older backend process that did not expose `/api/config`; the feature branch was started on `8081`. Avoid testing against a stale process.

### 2. Text ingestion worker ends `ingestion_failed`

`POST /api/ingestions/text` returned HTTP `202` in about 1.03 s, but the worker changed the job to:

```json
{
  "status": "failed",
  "stage": "failed",
  "error": {"code": "ingestion_failed", "message": "Unable to process ingestion"}
}
```

The public error is intentionally sanitized. Run the worker in the foreground and expose the underlying exception locally. If necessary, invoke the appropriate worker stage directly outside `process_job`, because `process_job` converts unexpected exceptions to the safe `ingestion_failed` error.

Check the complete stage path: claim and lease fencing, text chunking, Vertex embedding, MongoDB transaction/publication, and final stage update. Do not weaken lease checks or return raw exception text to API clients.

### 3. `app.check_config` incorrectly requires a Developer API key

`uv run python -m app.check_config` reports:

```text
Missing required settings: GEMINI_API_KEY
```

This is wrong when `GEMINI_PROVIDER=vertex_ai`. The required provider configuration should be conditional:

- Developer API: require `GEMINI_API_KEY`.
- Vertex AI: require `GOOGLE_CLOUD_PROJECT` or `GCP_PROJECT_ID`, plus `GOOGLE_CLOUD_LOCATION`; ADC must be available at runtime.

Keep the existing MongoDB and search-index checks.

## Recommended Resume Sequence

1. Confirm the branch and preserve `docs/test_document.md`.
2. Stop or avoid stale local API/worker processes; use known ports and foreground logs.
3. Reproduce the Top K `0` chat failure and capture the full server traceback.
4. Reproduce one text-ingestion job with the worker in the foreground and capture the internal exception.
5. Fix root causes on a new feature branch based on `feature/backend-v2-observability`; also fix conditional provider validation in `app/check_config.py`.
6. Add the smallest regression tests that fail for each root cause.
7. Run `uv run ruff check .` and `uv run pytest`.
8. Run backend QA. Fix and repeat QA until clear.
9. Start a clean API and worker and rerun every curl scenario below.
10. Append exact status codes, terminal states, latency, safe token metrics, and cleanup results to `docs/test_document.md`.
11. Commit and push the integration-fix branch for review. The project owner will merge branches into `develop`.

## Curl Validation Still Required

Use `docs/test_document.md` as the executable baseline. The final pre-merge run must cover:

- Health, login, invalid login, unauthenticated rejection, and `/api/auth/me`.
- Protected `/api/config`, including Top K and upload limits.
- Pasted-text ingestion, polling through stages, publication, listing, and deletion.
- Successful TXT, PDF, DOCX, CSV, and JSON upload/extraction through GCS and the worker.
- Invalid signature/content, malformed JSON/CSV, unsupported extension, and 20 MiB rejection before upload/job creation.
- Top K `0`, default Top K, explicit Top K, and invalid `21`.
- Hybrid retrieval returning ranked chunks and citations.
- Non-streaming and SSE chat, conversation continuation, and history persistence.
- Calculator tool activity in chat and SSE, followed by conversation reopen.
- Source chunk plus neighbors, format-aware location metadata, and authenticated original-file download.
- Conversation and document deletion, including GCS cleanup state.
- `chat_metrics`, `worker_stage`, `worker_job`, and `worker_cleanup` output with latency and token fields but no prohibited content.
- MongoDB vector and lexical index readiness through the corrected `app.check_config`.

## Known POC Limitations

- API and worker share one container and instance. Either process exiting restarts the container, and termination stops both.
- Cloud Run must keep CPU allocated for the worker loop.
- MongoDB Atlas 8.0+ and both configured search indexes are required.
- Atlas indexing is asynchronous; newly published chunks may take several seconds to appear in retrieval.
- The deployment topology is appropriate for this POC, not a production worker fleet.
- Non-stream generation cannot measure true first-token latency and reports it as unavailable; streaming measures the first emitted token.
- Token counts use provider metadata where supplied and estimates otherwise.

## Development Loop

For each new scope or fix:

1. Read the PRD and relevant docs; define features and action plan.
2. Show the complete action plan and obtain confirmation before coding.
3. Implement one task at a time.
4. Run backend QA after coding.
5. Fix findings and rerun QA until clear.
6. Review, commit, and push the feature branch; the project owner merges it.
7. Continue to the next task only after the previous loop is complete.

Do not commit `.env`, credentials, generated artifacts, or unrelated workspace changes.
