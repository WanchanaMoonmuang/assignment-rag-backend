# Backend QA Log

## 2026-07-13 — Full PRDv2 §11 acceptance validation (branch: fix/backend-v2-completion, commit: pending)

Environment: live services (MongoDB Atlas `poc_rag`, Vertex AI `gemini-3.5-flash` / `gemini-embedding-2` in `us` location, GCS `poc-rag-58`). Checks: `ruff check .` — all checks passed. `pytest` — 61 passed.

### Test cases

| ID | Scenario | Steps / request | Expected | Actual | Status |
| --- | --- | --- | --- | --- | --- |
| BQA-001 | Chat with retrieval disabled | `POST /api/chat {"top_k":0}` | 200, answer, no citations | 200, correct answer, `sources: []` | PASS |
| BQA-002 | Text ingestion end-to-end | `POST /api/ingestions/text` → poll → chat retrieval | job completes, chunk retrievable with citation | completed, cited correctly | PASS |
| BQA-003 | PDF ingestion + page citations | Upload real 2-page PDF (`fitz`-generated) | completes, page-range citations | completed (after 1 retry, see BUG-B-004), citations `Page 1`/`Page 2` correct | PASS (with caveat — see BUG-B-004) |
| BQA-004 | CSV ingestion + table-aware extraction | Upload CSV with headers + numeric rows | completes, summary + row-range chunks, numbers preserved | completed (after 1 retry, see BUG-B-004), `chunk_type: summary` with correct min/max/mean, `chunk_type: table` with row range, answerable via chat | PASS (with caveat — see BUG-B-004) |
| BQA-005 | JSON ingestion + record citations | Upload JSON array of 2 objects | completes, per-record citations | completed cleanly, `location: {type: record, path: "$[0]"/"$[1]"}` | PASS |
| BQA-006 | DOCX ingestion | Upload structurally valid `.docx` | completes, section citations | **fails every time** — see BUG-B-003 | FAIL |
| BQA-007 | `top_k` upper bound rejection | `POST /api/chat {"top_k":21}` | 422 | 422 with clear validation detail | PASS |
| BQA-008 | `top_k` at max valid value | `POST /api/chat {"top_k":20}` | 200, up to 20 chunks, multi-document | 200, 8 sources spanning PDF/JSON/TXT/CSV | PASS |
| BQA-009 | `top_k` omitted → default | `POST /api/chat` with no `top_k` | uses `RAG_TOP_K`=5 | 200, 5 sources returned | PASS |
| BQA-010 | No relevant content → general knowledge | Ask about a topic absent from ingested docs | answer, no forced legacy refusal | answered helpfully, described what IS available, no citations forced | PASS |
| BQA-011 | Non-English question | Ask in Spanish | answer follows question's language | correct, fluent Spanish answer with citations | PASS |
| BQA-012 | Multi-turn history | 2 chat turns in same `conversation_id`, second referencing first implicitly | follow-up answer uses context | correctly answered the implicit follow-up ("what about digital goods") | PASS |
| BQA-013 | Citation persistence on reopen | `GET /api/conversations/{id}` after a cited chat | identical sources returned, not re-retrieved | full source objects persisted verbatim | PASS |
| BQA-014 | Chunk + neighbor inspection | `GET /api/documents/{doc}/chunks/{chunk}` | chunk + neighbors + location | correct chunk, correct location, neighbors returned | PASS |
| BQA-015 | Original file requires auth | `GET /api/documents/{doc}/file` with no token | 401 | 401 `Not authenticated` | PASS |
| BQA-016 | Calculator, non-streaming, simple expression | `POST /api/chat {"question":"47*89"}` | correct result, tool visible | correct answer BUT empty `tool_activity` (inconclusive on its own) | PASS (answer) / see BUG-B-005 |
| BQA-017 | Calculator, non-streaming, hard expression | Large multiplication+subtraction requiring the tool | correct result, tool visible | **wrong answer**, empty `tool_activity` | FAIL — BUG-B-005 |
| BQA-018 | Calculator, SSE streaming | Same hard expression via `/api/chat/stream` | `tool_call`/`tool_result` events + final answer | tool events correct and exact, but stream **ends in `error` event, no final answer** | FAIL — BUG-B-005 |
| BQA-019 | Deprecated V1 `/api/ingest` compat | `POST /api/ingest` | 202, same job pipeline as V2 text ingestion | 202, job queued and completed normally | PASS |
| BQA-020 | Observability metrics content safety | Inspect `chat_metrics` JSON (handler attached manually for inspection) | latency/token/score fields present, no prompts/content/credentials | confirmed — see BUG-B-001 for why it's invisible by default | PASS (content) / see BUG-B-001 (emission) |
| BQA-021 | Idempotency (code review only — not exercised live) | Review `publish_job` transaction + job-lease claim filter | retry doesn't duplicate documents/chunks | `job_filter` requires matching `lease_token` on every update; `publish_job` inserts document + chunks + completes job in one transaction — a retried claim after a crash mid-transaction would re-run `publish_job` fully, and since chunk `_id`s are freshly generated (`make_id("chunk")`) on each call, a crash *after* the transaction commits but *before* the worker acknowledges could theoretically not duplicate (transaction is atomic), but a crash mid-`publish_job` before the transaction starts and a subsequent retry looks safe. No live duplicate observed in testing. | PASS (by review) |
| BQA-022 | Timeout handling (code review only) | Review `asyncio.wait_for(..., timeout=settings.ingestion_processing_timeout_seconds)` in `process_job` | timeout terminates + retries within limit | `app/worker.py` wraps processing in `asyncio.wait_for` and catches `TimeoutError` → `processing_timeout` error code, routed through the same `mark_job_failed` retry/terminal logic as other failures | PASS (by review) |
| BQA-023 | Full-file oversize rejection (413) | Code review of `create_file_ingestion`/`create_text_ingestion` | reject before conversion/upload | both endpoints check byte length before any GCS/worker interaction and raise 413 pre-emptively | PASS (by review; not exercised with an actual 20MiB+ payload this run) |
| BQA-024 | Content/extension mismatch (415) | Code review of `validated_content_type` | mismatched content → 415 | signature checks per format (PDF magic bytes, DOCX zip+`word/document.xml`, JSON parse, UTF-8 decode) all raise 415 on mismatch | PASS (by review) |

### Bugs

- **BUG-B-001** (severity: high, status: fixed) — Structured observability metrics (`chat_metrics`, `worker_stage`, `worker_job`, `worker_cleanup`) are never actually emitted in a running process.
  - Repro: start the API (`uv run uvicorn app.main:app --port <N>`) and worker (`uv run python -m app.worker`) exactly as documented (no special flags), drive any chat or ingestion request, and grep the process stdout for `chat_metrics`/`worker_stage` — nothing appears, even though the request succeeds.
  - Observed: `app/observability.py`'s `emit()` calls `logger.info(json.dumps(...))` against `logging.getLogger("app.metrics")`. Nothing in `app/main.py`, `app/worker.py`, or the `Dockerfile` CMD ever calls `logging.basicConfig()` or attaches a handler to the root logger. Python's root logger defaults to level `WARNING` with zero handlers, so every `INFO`-level `emit()` call is silently dropped. Confirmed by attaching a temporary `logging.basicConfig(level=logging.INFO)` in a throwaway harness script (not committed) — the exact same `emit()` calls then print correctly-shaped JSON (see below), proving the bug is purely missing logging configuration, not `emit()`'s logic.
  - Suspected root cause: missing `logging.basicConfig(...)` (or equivalent handler/level setup) in `app/main.py`'s app startup (e.g. `lifespan`) and `app/worker.py`'s `run()`.
  - Verified-safe payload shape once a handler is attached (from a real live chat request): `{"event":"chat_metrics","request_id":"req_...","conversation_id":"conv_...","provider":"vertex_ai","model":"gemini-3.5-flash","requested_top_k":5,"effective_top_k":5,"tool_names":[],"request_total_ms":3238.29,"status":"completed","retrieval_ms":1318.17,"prompt_build_ms":0.07,"model_first_token_ms":null,"model_total_ms":1439.07,"input_tokens":198,"output_tokens":25,"total_tokens":223,"token_source":"provider","retrieved_chunks":1,"scores":[0.0327...]}` — no prompts/answers/document content/snippets/credentials present. Content design is correct; only emission is broken.
  - Fix / re-verified: fixed in `app/observability.py` — the `app.metrics` logger now gets `setLevel(INFO)` + a dedicated `StreamHandler` (with `propagate = False` so other loggers are unaffected) at module load, so both `app/main.py` and `app/worker.py` pick it up automatically via their existing `from app.observability import emit` import — fixed once, at the single place `emit()` lives. Re-verified live: started a plain `uv run uvicorn app.main:app` with no special flags and drove a real chat request; `chat_metrics` appeared in stdout with the same safe shape as before, no code changes needed to see it.

- **BUG-B-003** (severity: high, status: fixed) — DOCX ingestion always fails; the `markitdown[docx]` optional dependency is not declared.
  - Repro: build any structurally valid `.docx` (proper `[Content_Types].xml`, `_rels/.rels`, `word/document.xml`) and call `app.extraction.extract(data, ".docx", 900, 150)` directly, or upload it via `POST /api/ingestions/file`.
  - Observed: `MarkItDown().convert(...)` raises `markitdown._exceptions.FileConversionException` wrapping `DocxConverter ... MissingDependencyException: ... include the optional dependency [docx] or [all] when installing MarkItDown ... pip install markitdown[docx]`. `app/extraction.py:258-260` catches this and re-raises as `ExtractionError("Document could not be converted")`, so the failure is clean (surfaces as a safe `extraction_failed` job error, no partial/corrupt data) — but DOCX ingestion, an explicit PRDv2 §4.4/§11.1 requirement, never succeeds regardless of input validity.
  - Suspected root cause: `pyproject.toml:10` declares `"markitdown>=0.1.6"` with no extras. Needs `"markitdown[docx]>=0.1.6"` (or `markitdown[all]`) so `python-docx`/related deps are installed.
  - Fix / re-verified: fixed — `pyproject.toml` now declares `"markitdown[docx]>=0.1.6"`; `uv sync --extra dev` installed `mammoth`/`lxml`/`cobble`. Re-verified live end-to-end: uploaded a real `.docx` via `POST /api/ingestions/file`, job reached `completed`, and a follow-up chat question correctly cited both the prose and table chunks from the document.

- **BUG-B-004** (severity: medium, status: fixed) — File ingestion has a race between job-claimability and GCS upload completion, plus the job's stale `error` field is never cleared on a successful retry.
  - Repro: `POST /api/ingestions/file` with a real PDF, then poll `GET /api/ingestions/{job_id}` until terminal, then inspect the raw job document.
  - Observed: job `job_11ffab9d2bad41f4af85b7fb8b458d89` (`sample.pdf`) ended with `status: completed`, `attempts: 2`, and a **populated** `error: {code: extraction_failed, message: "Original file could not be downloaded"}` — i.e. attempt 1 failed to download the object from GCS (raised in `process_file_job`, `app/worker.py` ~277-282), attempt 2 succeeded (chunks are correctly retrievable via chat with proper page citations), but the leftover `error` from attempt 1 was never cleared. `GET /api/ingestions/{job_id}` (`serialize_job`, `app/main.py:386-394`) returns `job.get("error")` unconditionally, so a client polling this endpoint sees `status: completed` alongside a populated `error` object — a confusing/contradictory contract, and arguably a §4.5 job-state-clarity violation (docs describe failed jobs as retaining a safe error, but say nothing about completed jobs also showing one).
  - Suspected root cause (two parts):
    1. Race: `create_file_ingestion` (`app/main.py` ~496-506) does `job_col(...).insert_one(job)` (making the job immediately claimable, status `queued`) BEFORE `await upload_object(...)` completes. The worker's poll loop (`app/worker.py`'s `run()`, ~1s idle poll) can claim the job and attempt `download_object` in that window, before the object actually exists in GCS — causing a spurious first-attempt failure that only self-heals via retry. Under real GCS latency or a busy worker, this could exhaust `INGESTION_JOB_MAX_ATTEMPTS` (default 3) and terminally fail a perfectly valid upload.
    2. Stale error: `publish_job`'s success path (`app/worker.py` ~237-249) `$set`s `status`/`stage`/`completed_at`/`updated_at` and `$unset`s lease fields, but never `$unset`s `error` from a prior failed attempt.
  - Fix / re-verified: fixed. `create_file_ingestion` (`app/main.py`) now defers `job_col(...).insert_one(job)` until after `upload_object` succeeds (on upload failure it inserts the job directly in a `failed` state instead), so the job is never claimable before the object exists in GCS. `publish_job`'s success update (`app/worker.py`) now also `$unset`s `error`. Re-verified live: uploaded a fresh PDF, job completed with `attempts: 1` and `error: None` (previously `attempts: 2` with a stale populated `error`).

- **BUG-B-005** (severity: high, status: fixed) — Calculator tool is unreliable on the non-streaming path and broken on the streaming path.
  - Repro (non-streaming): `POST /api/chat {"question":"Calculate 8492734 * 173829 - 5566 using the calculator tool.","top_k":0}`.
  - Observed (non-streaming): answer was `1,476,283,556,920`; ground truth (verified via Python) is `1,476,283,452,920` — **wrong**, confidently presented, with an intermediate step also wrong (`8492734 × 173829 = 1,476,283,562,486` vs true `1,476,283,458,486`). `tool_activity` was `[]` despite the server log showing `AFC remote call 1 is done.` / `AFC remote call 2 is done.` — i.e. Vertex's Automatic Function Calling DID invoke the `calculator` closure in `GeminiClient.generate_with_calculator` (`app/rag.py` ~214-238) twice, but the `activity` list returned to the caller stayed empty, so (a) the client can't see that a tool ran, and (b) whatever the tool actually computed was overridden or ignored by the model's final text, which is simply wrong. A simpler case (`47 * 89`) returned the correct answer with empty `tool_activity` too, but that one is easy enough a model could compute it unaided, so it doesn't prove the tool fired correctly — the harder case does prove it's broken.
  - Repro (streaming): `POST /api/chat/stream` with the same question.
  - Observed (streaming): `tool_call`/`tool_result` events fire correctly and the result is exactly correct (`"display_value": "1476283452920"` — matches ground truth), but the stream then ends with `event: error, data: {"message": "Unable to generate response"}` instead of a final answer. Server log shows the follow-up `generateContent` call (`app/rag.py` `stream_with_calculator`, ~303-311) returns `400 Bad Request` from Vertex, preceded by `Warning: there are non-text parts in the response: ['function_call']`. Suspected root cause: the follow-up call's `contents` list mixes a raw prompt string with explicit `Content` objects (`[prompt, Content(role="model", parts=call_parts), Content(role="user", parts=response_parts)]`) instead of a well-formed `Content(role="user", ...)` wrapper for the prompt — likely malformed multi-turn structure that Vertex rejects.
  - Root cause (confirmed via isolated repro before fixing): the bare-Python-callable "Automatic Function Calling" style used by `generate_with_calculator` never actually invoked the closure — confirmed by monkeypatching `calculate` and observing it was never called even while the SDK logged "AFC remote call N is done." The `stream_with_calculator` path's explicit `FunctionDeclaration` approach DID correctly invoke the tool, but its follow-up call failed with `400 Function call is missing a thought_signature` because it rebuilt `Part(functionCall=function_call)` from scratch instead of reusing the model's own returned content (which carries the required signature for this "thinking" model).
  - Fix / re-verified: fixed. Rewrote `generate_with_calculator` to use the same explicit `FunctionDeclaration`/manual-round-trip pattern as `stream_with_calculator` (shared via a new `_calculator_config`/`_calculator_response_part` helper), and both methods' follow-up calls now reuse the model's own returned content object (`response.candidates[0].content` for non-streaming; accumulated `chunk.candidates[0].content.parts` for streaming) instead of hand-building function-call parts. Re-verified live with a hard expression (`8492734 * 173829 - 5566`, ground truth `1476283452920` confirmed via Python): non-streaming now returns the correct answer with `tool_activity` populated; streaming now emits correct `tool_call`/`tool_result` events followed by a correct streamed final answer and a `done` event (previously ended in an `error` event with no answer). Added regression tests (`test_generate_with_calculator_reports_activity_and_preserves_model_content`, and an updated `test_stream_with_calculator_handles_multiple_calls`) asserting the follow-up call reuses the model's own content object.

- **BUG-B-002** (severity: none — false alarm, resolved) — `docs/handoff.md`'s two "runtime blockers" (`POST /api/chat {top_k:0}` → 500; text ingestion worker → `ingestion_failed`) do not reproduce on this branch. Root-caused live: they were caused by environment misconfiguration in this session's first setup attempt (`GOOGLE_CLOUD_LOCATION=us-central1` instead of the required `us`, and a credentials file path mismatch), not code defects. Once corrected, both flows work end-to-end (see BQA-001, BQA-002 below). Logged here only for continuity with the handoff doc; no code change needed.

## 2026-07-14 — Phase 3 gap-fix verification (branch: fix/backend-v2-completion)

All four bugs found in the 2026-07-13 validation run (BUG-B-001, BUG-B-003, BUG-B-004, BUG-B-005) are fixed and re-verified live, per the notes attached to each bug above. Summary:

- `uv run ruff check .` — all checks passed.
- `uv run pytest` — 62 passed (added `test_check_config_provider_validation_is_conditional`, `test_generate_with_calculator_reports_activity_and_preserves_model_content`; updated `test_stream_with_calculator_handles_multiple_calls`'s mock to include `candidates[0].content.parts`).
- Live re-verification (fresh API + worker, real MongoDB Atlas/Vertex AI/GCS): DOCX ingestion end-to-end with citations; PDF ingestion in exactly 1 attempt with no stale `error`; calculator correct on both `/api/chat` (with populated `tool_activity`) and `/api/chat/stream` (with `tool_call`/`tool_result` events followed by a correct answer and `done`); `chat_metrics` visible in plain stdout with no special configuration.
- One non-reproducible observation during re-verification: a long-running worker process (several hours old, having processed dozens of jobs across this session) intermittently failed GCS downloads that succeeded instantly from a fresh process/script. Restarting the worker resolved it every time. Not one of the four tracked bugs — no code fix applied — noting it here in case it recurs in a real long-lived deployment; possibly sandboxed-environment GCS client degradation under heavy iteration, not confirmed to be a production concern.

Test-case status updates from the prior run: BQA-003, BQA-004 no longer need the "with caveat" qualifier (race eliminated); BQA-006, BQA-017, BQA-018 flip from FAIL to PASS; BQA-020's emission caveat is resolved.

## 2026-07-14 — Full-loop PRD re-validation (branch: fix/backend-v2-completion)

Re-ran the full feature set against PRDv2 live (fresh MongoDB Atlas/Vertex AI/GCS, no
mocks), specifically exercising several items the prior run had only verified by code
review. Test IDs continue from BQA-024.

| ID | Case | Steps | Expected | Actual | Result |
|----|------|-------|----------|--------|--------|
| BQA-025 | Multi-document comparison (live, not code review) | Ingested two short documents with distinct comparable facts (drone weight/price/flight-time), asked a comparison question via `/api/chat` with no `document_id` filter | answer compares both, citations reference both source documents (PRDv2 §11: "a comparison fixture retrieves relevant chunks from at least two documents ... without document selection") | answer correctly stated both drones' weight/price/flight-time with `[1]`/`[2]` citations mapping to the two distinct `document_id`s; no manual selection used | PASS |
| BQA-026 | Idempotency / crash-recovery (live, not code review) | Queued a job, then directly set it to `status: processing`, `stage: chunking`, `attempts: 1`, an **expired** `lease_expires_at` (simulating a worker that claimed the job and crashed mid-processing) via a MongoDB write; started a fresh worker | fresh worker reclaims via the `processing` + expired-lease branch of `claim_job`'s filter, completes at `attempts: 2`, exactly one set of chunks (no duplicates from the "crashed" first attempt) | worker log showed `attempt=2`, full stage pipeline completed; chunk collection had exactly 1 chunk for that document | PASS |
| BQA-027 | Ingestion timeout handling | Existing automated regression test `tests/test_app.py::test_worker_records_processing_timeout` — monkeypatches `process_text_job` to outlast `ingestion_processing_timeout_seconds`, asserts `error.code == "processing_timeout"` | timeout raises inside `process_job`'s `asyncio.wait_for`, routes through the same failure/retry path as other errors | test passes (part of the 62-test suite); exercises the real `process_job` code path, not just a read-through | PASS — upgraded from BQA-022's "code review only" |
| BQA-028 | Document deletion + GCS cleanup lifecycle (live) | Ingested a document, then `DELETE /api/documents/{id}` | chunks deleted, document deleted, GCS original object deleted (`cleanup_completed`), re-downloading the object 404s | all confirmed: chunks gone, document gone, GCS object gone, cleanup job terminal at `cleanup_completed` | PASS |
| BQA-029 | Reject oversized/malformed/mismatched ingestion payloads (live, not code review) | Oversized text, oversized file, fake-signature "PDF", malformed JSON, malformed CSV, unsupported `.exe` extension, each via the real endpoints | 413 for size, 415 for signature/format mismatches, clean `extraction_failed` (not a crash) for structurally-invalid-but-accepted formats like CSV | oversized text → 413; oversized file → 413; fake PDF signature → 415; malformed JSON → 415 at upload; malformed CSV → 202 at upload then a clean worker-stage `extraction_failed` ("CSV rows must match the header columns"); `.exe` → 415 | PASS — upgraded from BQA-023/BQA-024's "code review only" |

### Observability re-investigation (BUG-B-001 correction)

The 2026-07-13 fix for BUG-B-001 (missing `logging` handler in `app/observability.py`) was
only cleanly re-verified live for the **API** process in that session; this round set out
to confirm the **worker** process's `worker_stage`/`worker_job` events the same way, and
initially could not reproduce reliable output — plain `uv run python -m app.worker` runs
showed zero log lines even though jobs completed correctly every time, across roughly a
dozen different hypotheses (stream target, line-buffering, `PYTHONUNBUFFERED`, a startup
probe line, swallowed-exception checks).

**Root cause, found via a `/proc` scan (not `ps aux`, which reported nothing running):**
this long session had accumulated **~40 leftover `uv run python -m app.worker` and
`uvicorn` processes** from earlier test rounds, all still alive and polling MongoDB. Job
claims are atomic (`find_one_and_update`), so whichever stray instance won the race
processed the job — not necessarily the one actually being tested — which explains the
inconsistent, seemingly buffering-related symptoms across many "fixes." None of those
fixes were the real cause. Killing every stray process and re-testing with exactly one
worker instance showed the original fix works correctly and deterministically: every
`worker_stage`/`worker_job` event for a real job appears in stdout, in order, every time.

- **Genuine fix kept and committed** (commit `94e3688`): `app/observability.py` now
  explicitly binds `logging.StreamHandler(stream=sys.stdout)` (the previous version used
  the bare `StreamHandler()` default, which is `stderr`) and force-enables line-buffering
  on `sys.stdout`. This is correct regardless of the process-leak issue above and is a
  reasonable safeguard for Cloud Run's stdout-only capture.
- **No code change was needed or made for the worker-specific "invisible logs" symptom**
  — it was a test-environment artifact (leftover background processes from a long
  session), not a defect in `app/worker.py` or `app/observability.py`. `docs/handoff.md`
  and this file are corrected accordingly: BUG-B-001 is fixed for both API and worker
  processes, confirmed live for both.
- Practical takeaway for future long debugging sessions in this environment: `ps aux`
  can fail to show processes started via `&`/`disown` in earlier shell invocations even
  though they're still alive; scan `/proc/*/cmdline` directly and kill stragglers before
  trusting a "flaky" result.

### Cleanup

All test/debug documents created during this round and the observability investigation
(19 total, e.g. `zephyr-x200.txt`, `idempotency-test.txt`, various `*-probe.txt`/
`*-test.txt` artifacts) were deleted via `DELETE /api/documents/{id}` after validation;
`GET /api/documents` confirmed the collection is empty of test artifacts. All test API
server and worker processes for this round were killed after use.

`uv run ruff check .` — all checks passed. `uv run pytest` — 62 passed (unchanged from
Phase 3; no test code needed to change for this round's findings).

## 2026-07-14 — Bug-fix batch verification: calculator scope, PDF table dedup, tool_activity persistence (branch: fix/bug-batch-backend, commit: 9b90ab5 + uncommitted working tree)

Environment: live services (MongoDB Atlas, Vertex AI `gemini-3.5-flash`, GCS), API on
`:8080` + worker as two foreground processes, plus unit-only static checks. Checks: `uv run
ruff check .` — all checks passed (0 issues). `uv run pytest` — 64 passed, 0 failed
(baseline 60; +2 new regression tests in this diff, +2 pre-existing from prior rounds).
`uv run python -m app.check_config` — `Backend config ok` (used corrected
`GOOGLE_APPLICATION_CREDENTIALS=/home/developer/.config/gcloud/gcp_key.json`, per the known
local env quirk — not a code issue).

Diff under test: `app/extraction.py`, `app/main.py`, `app/rag.py`, `tests/test_app.py`,
`tests/test_extraction.py` (uncommitted, working tree only).

### Test cases

| ID | Scenario | Steps / request | Expected | Actual | Status |
| --- | --- | --- | --- | --- | --- |
| BQA-030 | B1: conceptual question does not trigger calculator (non-stream) | `POST /api/chat {"question":"What is a RAG pipeline?","top_k":0}` | 200, answer, `tool_activity: []` | 200, correct conceptual answer, `tool_activity: []` | PASS |
| BQA-031 | B1: conceptual question does not trigger calculator (SSE) | `POST /api/chat/stream {"question":"What is retrieval augmented generation used for?"}` | no `tool_call`/`tool_result` events | 0 tool events in the SSE stream; answer streamed and completed normally | PASS |
| BQA-032 | B1: explicit arithmetic still triggers calculator (non-stream) | `POST /api/chat {"question":"what is 47 * 6?","top_k":0}` | 200, answer "282", `tool_activity` populated | answer `"47 * 6 is 282."`, `tool_activity: [{name: calculator, arguments: {expression: "47 * 6"}, result: 282}]` | PASS |
| BQA-033 | B3: tool call available at `top_k=0`, non-streaming | same as BQA-032 (top_k explicitly 0) | tool still resolves despite retrieval disabled | confirmed — no gate on tool availability tied to `top_k` | PASS |
| BQA-034 | B3: tool call available at `top_k=0`, streaming | `POST /api/chat/stream {"question":"what is 12 * 9?","top_k":0}` | `tool_call` → `tool_result` → token(s) → `done`, correct result | SSE showed exactly that sequence, `tool_result` `display_value: "108"`, final answer `"12 * 9 is 108."` | PASS |
| BQA-035 | B6: persisted `tool_activity` never contains a dangling "requested" entry (live) | Same request as BQA-034, then `GET /api/conversations/{id}` | persisted assistant message's `tool_activity` has exactly one entry, `status: "completed"`, with `name`/`arguments`/`display_value` | confirmed: one entry, `{"name":"calculator","arguments":{"expression":"12 * 9"},"status":"completed","display_value":"108"}` — no `"requested"` entry present | PASS |
| BQA-036 | B6 unit regression exercises the real persistence branch | `tests/test_app.py::test_stream_chat_persists_only_resolved_tool_activity` | asserts `app/main.py`'s `stream_chat` only appends `tool_result` events to `tool_activity`, using a `FakeGemini` + real `TestClient` POST + inspecting the stored conversation doc (not the `GeminiClient` in isolation) | test does exercise the `app/main.py` persistence branch (posts through `/api/chat/stream`, reads back `app.state.db["conversations"]`), not just `stream_with_calculator` — genuine regression coverage | PASS |
| BQA-037 | B6: non-streaming persisted shape (`{name, arguments, result}`, no `status`/`display_value`) still renders correctly on the frontend | Code review: `ChatWorkspace.tsx`'s `ToolActivityRow` — `pending = status === "requested"`, `failed = status === "failed" \|\| Boolean(error)`, `value = display_value ?? (typeof result is primitive ? String(result) : undefined)` | missing `status` on the non-stream shape must not be interpreted as "pending" or break rendering | confirmed self-consistent: with `status` absent, `pending` and `failed` both evaluate `false`, and `value` falls back to `result` — renders as a completed line, not stuck spinning. This behavior predates the current diff (non-stream path/shape untouched by it) and is not made worse or better by it | PASS (no regression) |
| BQA-038 | B2/B5: PDF table-cell duplication regression test is a real assertion, not tautological | Reverted `app/extraction.py` only (`git stash push -- app/extraction.py`), reran `tests/test_extraction.py::test_pdf_extraction_does_not_duplicate_table_text_as_prose` | test fails pre-fix (proves the assertion is meaningful) | failed with `assert not True` (`first_row` from a table chunk found substring-inside a prose chunk) before the fix; passes after `git stash pop` restored the fix | PASS (test is a genuine regression guard) |
| BQA-039 | B2/B5: no table-row duplication in prose chunks across the stress-test PDF | `extract_pdf(samples/sample-tables.pdf)` (11 pages, up to 4 tables/page, 29 table chunks total) — checked every table row (not just first row) against every prose chunk | zero substring matches of any table row (len ≥ 8) inside any prose chunk | 40 total chunks (29 table, 11 prose), 0 duplicate rows found | PASS |
| BQA-040 | B2/B5: heuristic over-exclusion check (does dropping overlapping blocks lose legitimate nearby prose?) | For every page with tables, dumped every `page.get_text("blocks")` block classified as "excluded" (overlaps a table bbox) and inspected its text | excluded blocks should be genuine table cell content, not incidental prose/captions | manually inspected all excluded blocks across all 11 pages of `sample-tables.pdf` — every excluded block was table header/data-cell text; no prose paragraph or caption was misclassified as table content in this file | PASS (for this file; see note below on residual heuristic risk) |
| BQA-041 | B4 (question, no fix) — confirm no code change and no regression | Diff review of `app/extraction.py`'s DOCX path | `extract` still routes `.docx` through `extract_markdown()`/MarkItDown, unchanged | confirmed no code change in this diff touches DOCX routing; matches the answer already given to the user (intentional, generic markdown path, name is just imprecise) | PASS (no-op, as expected) |
| BQA-042 | Full regression sweep | `uv run ruff check .`, `uv run pytest` | 0 lint issues, all tests pass | 0 issues; 64 passed | PASS |
| BQA-043 | `chat_metrics` content safety spot-check during this run's live calls | Grepped API stdout for `chat_metrics` events emitted by BQA-030/032/033/034 requests | latency/token/tool_names fields present, no chat text/document content/credentials | confirmed: fields present (`retrieval_ms`, `model_total_ms`, `input_tokens`/`output_tokens`, `tool_names`), no prohibited content in any event this run | PASS |
| BQA-044 | B2/B5 collateral-change check: the prose-assembly mechanism itself switched from `page.get_text("text")` to a `page.get_text("blocks")` join **on every PDF page, including pages with zero tables** — this affects all PDF ingestion, not just table dedup. Verified it doesn't garble reading order or splice in non-text (e.g. image) block content on two table-free samples (`samples/MKT556._outline.pdf`, `samples/Content_Strategy_Plan_ODOT.pdf`) | For every page of both files: compared old `page.get_text("text")` output against the new `blocks`-join output (whitespace-normalized), and checked for any block with a non-zero block-type (image/non-text) | identical content and order between old and new extraction mechanisms; no non-text blocks present | exact match (whitespace-normalized) on every page of both files (2 pages + 7 pages); zero non-text blocks found in either file | PASS |

### Bugs

No new bugs found in this batch. All five acceptance criteria (B1, B2/B5, B3, B6; B4 is a
question with no fix, confirmed as such) verified PASS, live where credentials allowed and
by code review/unit test otherwise. No previously logged bug (BUG-B-001 through BUG-B-005)
regressed — full pytest/ruff sweep and live calculator + observability spot-checks came back
clean.

### Note (not a bug — heuristic risk worth tracking)

The PDF dedup fix in `app/extraction.py`'s `extract_pdf`/`_overlaps` excludes a text **block**
(from `page.get_text("blocks")`) from the prose pass if it overlaps a detected table's bbox
**at all** (any non-zero intersection), i.e. block-granularity, all-or-nothing, no partial-
overlap threshold. On `samples/sample-tables.pdf` this was empirically safe — every excluded
block was genuine table content, and zero row-level duplicates remained (BQA-039/040). Two
theoretical edge cases were not reproducible with the available sample files and are worth a
future stress test if a bug report ever surfaces on a different PDF layout:
1. **Over-exclusion**: a caption or prose paragraph whose block happens to sit inside a
   table's detected bbox (e.g. a slightly generous `find_tables()` bbox, or a caption
   immediately above/below a table absorbed into the same visual block) would be dropped
   entirely, silently losing that prose from retrieval.
2. **Under-exclusion**: exclusion is block-granularity and all-or-nothing (any bbox
   intersection drops the whole block) — the real gap is `find_tables()` itself
   under-detecting a table's true rendered extent (e.g. rotated/merged cells, unusual
   multi-column layouts). If the detected bbox doesn't cover the actual table area, the
   corresponding text block(s) won't intersect it at all and won't be excluded, so their
   content would still duplicate the table chunk — the fix's protection is only as good as
   `find_tables()`'s own bbox accuracy.
Neither was observed; flagging only because the acceptance criteria explicitly asked for
this consideration and no PDF exercising these layouts was available in `samples/`. Also
verified (BQA-044) that swapping `get_text("text")` for a `get_text("blocks")` join — which
runs on every page, not only pages with tables — does not change extracted content or
reading order on two table-free sample PDFs, and introduces no non-text block content.

### Cleanup

Live test conversations created during this round (BQA-030–034) were left in MongoDB
Atlas (chat/document text is not persisted anywhere in this log, per the redaction rules);
no test documents were ingested this round (no ingestion changes were in scope for this
batch), so there was nothing to delete via `DELETE /api/documents/{id}`. API and worker
foreground processes started for this round were killed after use; `/proc` scan confirmed
no stray `uvicorn`/`app.worker` processes remained.

## 2026-07-18 — Chained calculator tool calls + eval report rewrite (branch: feat/ragas-eval, commit: e89e37c, uncommitted diff)

Environment: live services (MongoDB Atlas, Vertex AI `gemini-3.5-flash` / `gemini-embedding-2`,
GCS). Checks: `ruff check .` — all checks passed. `pytest` — 73 passed (baseline expectation
met exactly). `python -m app.check_config` — "Backend config ok".

Scope: uncommitted working-tree diff (not yet committed) touching `app/rag.py`,
`app/settings.py`, `.env.example`, `evals/run_eval.py`, `tests/test_app.py`, and new
`evals/test_report.py`. This QA reviewed that diff against the acceptance criteria handed
down by the coordinator (areas A/B/C below); no code was modified by this QA pass.

### Test cases

| ID | Scenario | Steps / request | Expected | Actual | Status |
| --- | --- | --- | --- | --- | --- |
| BQA-045 | A1: new chained/round-limit calculator tests pass | `pytest tests/test_app.py -k calculator -v` | 7 passed, incl. 4 new tests | 7 passed: `test_calculator_supports_safe_math_and_rejects_code`, `test_stream_with_calculator_handles_multiple_calls`, `test_stream_with_calculator_handles_chained_tool_calls`, `test_stream_with_calculator_stops_at_round_limit`, `test_generate_with_calculator_reports_activity_and_preserves_model_content`, `test_generate_with_calculator_handles_chained_tool_calls`, `test_generate_with_calculator_stops_at_round_limit` | PASS |
| BQA-046 | A1: the 4 new tests are genuine regressions, not tautological | `git stash push -u -- app/rag.py app/settings.py` (reverts to pre-diff single-round logic, keeps new tests), rerun `pytest -k "chained_tool_calls or round_limit"`, then `git stash pop` to restore the fix | all 4 new tests fail against the old single-round code | all 4 failed as expected: chained tests got `answer == ''` instead of the follow-up chained result (old code only reads `.text` off one follow-up, which is empty when that follow-up is itself a function call); round-limit tests recorded only 1 activity entry instead of 2 (no bound loop existed to hit a limit). Stash popped cleanly, `pytest -q` re-confirmed 73 passed afterward | PASS |
| BQA-047 | A2: round-limit tests prove the loop is actually bounded | Read `test_generate_with_calculator_stops_at_round_limit` / `test_stream_with_calculator_stops_at_round_limit` bodies — mock `Models` never stops returning `function_calls` | asserts `model_calls`/round count == `max_tool_rounds` (2, set explicitly in the test), not just "produces some answer" | confirmed: `assert len(model_calls) == 2` (generate) and `assert len(model_calls) == 2` (stream) with `max_tool_rounds=2` passed into `_calculator_client`; loop terminates instead of looping on an infinite-function-call mock | PASS |
| BQA-048 | A3: live multi-step calculator question, non-stream `/api/chat` | `POST /api/chat {"question":"What is (12 * 7) + (100 / 4)? ... Show each step."}` (top_k default) | ≥2 chained calculator calls in `tool_activity`, real computed answer, not `FALLBACK_ANSWER` | 200 OK; `tool_activity` shows 3 chained calculator calls (`12 * 7`→84, `100 / 4`→25.0, `84 + 25`→109); answer text states "The final result is **109**." — correct and not the fallback string | PASS |
| BQA-049 | A3: live multi-step calculator question, SSE `/api/chat/stream` | `POST /api/chat/stream {"question":"What is (18 * 3) + (144 / 12)? ... Show each step."}` | multiple `tool_call`/`tool_result` event pairs, correct final answer via `token` events, terminal `done` | SSE emitted 3 `tool_call`→`tool_result` pairs (`18*3`=54, `144/12`=12.0, `54+12`=66), token stream concluded "The final result is **66**.", followed by `sources` and `done` events | PASS |
| BQA-050 | A4: existing non-chained/batched-call behavior unchanged | `test_stream_with_calculator_handles_multiple_calls`, `test_generate_with_calculator_reports_activity_and_preserves_model_content` | pass unmodified in behavior (only boilerplate refactored into `_calculator_client` helper) | both pass; diff review confirms only the `Part`/`client` construction boilerplate was extracted into the new `_calculator_client(models, max_tool_rounds=4)` helper — no assertion lines in either test were touched | PASS |
| BQA-051 | Observability: `chat_metrics` events from the live BQA-048/049 calls carry no prohibited content | Grepped API stdout for `chat_metrics` after both live calls | latency/token fields present, no chat text/answer/document content | confirmed: fields present (`retrieval_ms`, `model_total_ms`, `input_tokens`/`output_tokens`/`total_tokens`, `tool_names: ["calculator","calculator","calculator"]`, `scores`), no answer text, no calculator expression values, no document content in either event | PASS |
| BQA-052 | B1: `answer_question()` substitutes `FALLBACK_ANSWER` instead of returning blank | Code review: `evals/run_eval.py::answer_question` | `return chunks, answer or FALLBACK_ANSWER` (imported from `app.rag`), mirroring `app/main.py`'s `completed_answer`/`stream_chat` | confirmed present; `FALLBACK_ANSWER = "I could not generate an answer. Please try again."` in `app/rag.py:15` | PASS |
| BQA-053 | B2: `run()` computes per-row `sources` + deterministic `source_hit_rate` correctly | Code review: `evals/run_eval.py::run()`, the `sources`/`hits`/`source_hit_rate` block | `sources` = one `{document_name, location, score}` dict per retrieved chunk; `source_hit_rate` = fraction of those chunks whose `document_name` equals the golden question's own `filename` key; `0.0` when no chunks retrieved (no div-by-zero) | confirmed correct: `location` reads `chunk.get("location") or {}).get("label", "")` which matches the real chunk shape observed live in BQA-048 (`location: {type, start, end, label}`); `hits = sum(1 for source in sources if source["document_name"] == filename)`; guarded `sources / len(sources) if sources else 0.0` | PASS |
| BQA-054 | C1: `evals/test_report.py` passes explicitly, and is excluded from the default `pytest` run | `pytest evals/test_report.py -v` and separately `pytest -q` (no `--extra eval`) | `test_report.py` passes on its own; default run (73 tests) unaffected, no pandas import error | `evals/test_report.py::test_write_report_formats_metrics_and_citations` — 1 passed; `pyproject.toml`'s `testpaths = ["tests"]` confirmed to exclude `evals/` from the default collection — default `pytest -q` run (73 passed) does not import pandas via this file | PASS |
| BQA-055 | C2: `write_report()` output is well-formed Markdown (table columns, escaping, NaN, metric labels, blockquotes) | Ran `write_report()` directly against a synthetic `_FakeResult`/rows fixture (mirroring `evals/test_report.py`, plus one row with a model answer containing `#`/`##`/`\|` to stress the blockquote escaping) and read the generated `.md` and `.json` | header/separator/data table column counts match; no unescaped `\|` breaks table structure; NaN scores render `N/A` never `nan`; metric names are human-readable (`Faithfulness`, `Context Precision`, `Context Recall`, `Source Hit Rate`), not raw RAGAS columns; Model's Answer / Ground Truth are `> `-blockquoted so embedded Markdown headings don't break report structure; JSON output is valid | All confirmed as expected. Summary table header (`\| # \| File \| Source Hit Rate \| Faithfulness \| Context Precision \| Context Recall \|`) and its separator row both have 6 columns; `_fmt_score` correctly renders the injected `math.nan` faithfulness cell as `N/A` and `"nan"` never appears (case-insensitive check); `_metric_label` maps `llm_context_precision_with_reference`→`Context Precision` etc.; Model's Answer text containing `## Fake subheading` and `- bullet \| pipe` rendered fully inside `> `-prefixed blockquote lines, not as live Markdown structure; `.json` output parsed cleanly with `json.tool`. One gap found — see BUG-B-006 below (low severity, pre-existing, out of scope for this diff) | PASS (see BUG-B-006 caveat) |
| BQA-056 | Full regression sweep for this diff | `ruff check .`, `pytest -q`, `python -m app.check_config` | 0 lint issues, 73 passed, config ok | 0 lint issues; 73 passed (matches the acceptance criteria's expected count exactly, up from the 60-passed baseline mentioned in the general QA brief); `check_config` printed "Backend config ok" | PASS |

### Bugs

- **BUG-B-006** (severity: low, status: open, pre-existing/out-of-scope) — `write_report()`'s
  per-question `**Question:**` line is not blockquoted or pipe-escaped, unlike the adjacent
  `Model's Answer`/`Ground Truth` fields that this diff explicitly hardened.
  - Repro: call `write_report()` with a row whose `question` contains a Markdown heading,
    e.g. `"What is the total?\n# Injected heading\n| a | b |"` — inspect the generated
    `.md`; the `# Injected heading` line renders as a live H1 in the output report, and the
    `| a | b |` renders as stray table-like text outside any table context, both breaking
    the document's intended heading hierarchy.
  - Observed: `evals/run_eval.py:463`, `md_lines.append(f"**Question:** {row['question']}")`
    — no `_blockquote()`/`_escape_cell()` call, unlike the `answer`/`ground_truth` lines
    a few lines below it.
  - Suspected root cause: `evals/run_eval.py:463` — this exact line is unchanged from the
    pre-diff version (`git show HEAD:evals/run_eval.py` has the identical line), so it
    predates this diff and is not something the current acceptance criteria (which scoped
    the blockquote hardening specifically to "Model's Answer / Ground Truth") asked to fix.
    Risk is low in practice since `question` comes from the human-curated golden dataset
    (`samples/*.json` golden questions), not live model output, but it's the same class of
    issue the diff otherwise fixed nearby.
  - Fix / re-verified: pending — hand back only if the coordinator wants Question hardened
    to match; not blocking for this diff's stated acceptance criteria.

### Cleanup

Live test conversations created by BQA-048/049 were left in MongoDB Atlas (chat text is not
persisted anywhere in this log, per the redaction rules); no documents were ingested this
round (out of scope for this diff), so nothing to delete via GCS/`DELETE /api/documents/{id}`.
API (`uvicorn app.main:app --port 8080`) and worker (`python -m app.worker`) foreground
processes started for this round were killed after use; `/proc` scan confirmed no stray
`uvicorn`/`app.worker` processes remained. Scratch files (sample eval report, chat
response/stream captures, bearer token) were written only to the session scratchpad
(`/tmp/claude-*/.../scratchpad/`), never into the repo.
