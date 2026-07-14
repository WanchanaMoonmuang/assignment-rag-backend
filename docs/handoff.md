# Backend Handoff

Last updated: 2026-07-14

## Current State

Backend V2 is complete and validated against `docs/PRDv2.md` §11 acceptance criteria, live
against real MongoDB Atlas, Vertex AI, and GCS — including a full-loop re-validation pass
that live-tested multi-document comparison, idempotency/crash-recovery, and document
deletion + GCS cleanup (previously only verified by code review). All prior blockers are
resolved. **Next up: frontend V2** (per `PRDv2.md`) — the frontend currently remains on V1
behavior and this backend stays compatible with it (see V1-compat notes below).

Repository: `poc-rag-backend`

Current branch: `fix/backend-v2-completion` (based on `develop`, not yet merged/pushed)

Current committed head:

```text
94e3688 Emit worker/API metrics to stdout explicitly and line-buffer
```

`develop` (`bc8c198`) already contains the full V2 feature stack (PRs #4–#8: extraction,
hybrid retrieval, citations/tools, observability) — no branch-merge work remains.

## Source Documents

- Shared V2 product contract: `../docs/PRDv2.md`
- Shared architecture: `../docs/app-architecture.md`
- Shared metadata strategy: `../docs/rag_metadata_strategy.md`
- Backend setup and API summary: `README.md`
- Full QA log (test cases + bugs, incremental): `docs/QA.md`
- Prior curl scenarios and historical results: `docs/test_document.md`
- Search index definitions: `docs/mongodb-vector-index.json` and `docs/mongodb-search-index.json`

## What happened this session

The three "runtime blockers" recorded in the previous version of this doc (chat 500 on
`top_k:0`; worker `ingestion_failed`; `check_config` requiring the wrong provider key) were
investigated live. Two of them (the chat 500 and the `ingestion_failed`) turned out to be
**environment misconfiguration in this session's own setup**, not code bugs — see
"Required Local Configuration" below for the exact values that matter. Only the
`check_config` issue was a real code bug, and it's fixed.

A full live validation against every PRDv2 §11 acceptance criterion (all 24 test cases
in `docs/QA.md`) then found **four real bugs**, all fixed and re-verified live this
session:

1. **Observability metrics never emitted** — no logging handler was configured anywhere,
   so `chat_metrics`/`worker_stage` JSON never reached stdout despite the README's claim.
   Fixed in `app/observability.py`.
2. **DOCX ingestion always failed** — `markitdown[docx]` extra wasn't declared, so the
   conversion dependency was missing. Fixed in `pyproject.toml` (+ `uv.lock`).
3. **File-ingestion race + stale error** — the job became claimable before its GCS upload
   completed (spurious first-attempt failure, self-healing but wasteful), and a job's
   `error` field wasn't cleared on a successful retry. Fixed in `app/main.py` (defer job
   insert until upload succeeds) and `app/worker.py` (`$unset` error on success).
4. **Calculator tool broken** — the non-streaming path's bare-callable "Automatic Function
   Calling" never actually invoked the tool (confirmed by monkeypatching); the streaming
   path's tool call worked but its follow-up answer failed with a Vertex 400 (missing
   `thought_signature`, a Gemini 3.x "thinking" model requirement) because the follow-up
   rebuilt the function-call part from scratch instead of reusing the model's own returned
   content. Fixed in `app/rag.py` — both paths now share a helper and reuse the model's own
   content object in their follow-up calls.

Full repro steps, root causes, and re-verification evidence for all four are in
`docs/QA.md` (BUG-B-001, BUG-B-003, BUG-B-004, BUG-B-005). `uv run ruff check .` passes;
`uv run pytest` — 62 passed (up from 60 at last handoff; added/updated regression tests
for each fix).

A subsequent full-loop re-validation pass live-tested three items the first pass had only
verified by code review: multi-document comparison (a comparison question with no
document filter correctly cited two distinct documents), idempotency/crash-recovery
(simulated a crashed worker via an expired lease; a fresh worker reclaimed and completed
with no duplicate chunks), and the document deletion + GCS cleanup lifecycle end-to-end.
Ingestion timeout handling is covered by the existing `test_worker_records_processing_timeout`
regression test, which exercises the real `asyncio.wait_for` path in `process_job`.

This pass also chased down and resolved a false alarm in the worker process's
`worker_stage`/`worker_job` logging, which initially looked broken again after the first
pass's fix. Root cause: ~40 leftover worker/API processes from earlier in the same long
debugging session were still alive (invisible to `ps aux`, found via a `/proc` scan) and
racing to claim jobs, so logging visibility depended on which stray process won each
race — not a real defect. A small genuine improvement was kept (`app/observability.py`
now binds its handler explicitly to `sys.stdout` with line-buffering, since the previous
version relied on `StreamHandler()`'s default, which is `stderr`); no other code change
was needed. Full detail in `docs/QA.md`'s "Observability re-investigation" section.

## Main Modules

- `app/main.py`: FastAPI routes, auth-protected API orchestration, chat, SSE, ingestion submission, documents, conversations, and citations.
- `app/rag.py`: Gemini clients, prompts, token usage, streaming, calculator orchestration, and RAG helpers.
- `app/worker.py`: MongoDB job claiming, leases, extraction/embedding/publication stages, failure handling, and GCS cleanup sweep.
- `app/extraction.py`: format-specific extraction and chunk metadata.
- `app/storage.py`: GCS upload, download, and deletion.
- `app/calculator.py`: restricted arithmetic evaluator; does not use `eval`.
- `app/observability.py`: best-effort JSON event emission to stdout (now actually configured — see fix #1 above).
- `app/settings.py`: validated environment settings.
- `app/check_config.py`: MongoDB, Atlas index, and provider-conditional validation.
- `tests/test_app.py` and `tests/test_extraction.py`: backend coverage (62 tests).

## Required Local Configuration

Use `.env` for local runtime values but never commit it (already gitignored). Important
settings, with the two corrections found this session called out:

```text
GEMINI_PROVIDER=vertex_ai
GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID
GOOGLE_CLOUD_LOCATION=us          # NOT us-central1 — the configured models
                                   # (gemini-3.5-flash, gemini-embedding-2) 404 in
                                   # us-central1 for this project; us/global work.
MONGODB_URI
MONGODB_DATABASE=poc_rag           # this DB's two Atlas Search indexes are set up
                                   # and READY (document_chunks_vector_index,
                                   # document_chunks_text_index) — don't recreate them
MONGODB_VECTOR_INDEX
MONGODB_SEARCH_INDEX
GCS_BUCKET_NAME=poc-rag-58
JWT_SECRET_KEY
AUTH_USERNAME=admin
AUTH_PASSWORD
```

`GOOGLE_APPLICATION_CREDENTIALS` must point to a real, readable service-account key file
(watch for path typos — this session's session-level env var pointed at a hyphenated
filename that didn't exist; the real file used an underscore). Verify with:

```bash
uv run python -m app.check_config   # should print "Backend config ok"
```

## V1 Frontend Compatibility (do not break)

The frontend is intentionally still on V1 and depends on:
- Chat request field is `question` (PRDv2 §5.7 documents it as `message` — that's a **doc
  slip**, not an intended API change; the frontend and all backend code use `question`).
- `POST /api/ingest` (deprecated) must keep delegating to the V2 text-ingestion job.
- Legacy V1 documents/conversations remain listable, searchable, and deletable.

## Known POC Limitations (unchanged)

- API and worker share one container and instance in the POC Cloud Run topology. Either
  process exiting restarts the container; termination stops both.
- Cloud Run must keep CPU allocated for the worker loop (always-on, min/max instances 1).
- MongoDB Atlas 8.0+ and both configured search indexes are required.
- Atlas indexing is asynchronous; newly published chunks may take several seconds to
  appear in retrieval.
- Non-stream generation cannot measure true first-token latency; streaming measures it.
- Token counts use provider metadata where supplied and estimates otherwise.
- One informational, non-reproducible observation from this session: a long-running
  worker process occasionally had transient GCS download failures that a fresh process
  didn't reproduce. Not root-caused (didn't recur after restart); not one of the four
  tracked bugs. Worth a second look if it recurs in a real long-lived deployment.

## Next Steps

1. **Commit is done; push is pending user confirmation** (per project convention, always
   ask before pushing).
2. Once pushed, open a PR for `fix/backend-v2-completion` → `develop` for the project
   owner to review and merge.
3. **Then: frontend V2**, per `PRDv2.md` §7 (Frontend Requirements) and the rest of the
   contract — Top K slider from `/api/config`, source-inspection drawer with page/section
   highlighting, calculator tool-activity UI, citation persistence rendering, etc. The
   backend's V2 API contract (SSE `metadata`/`tool_call`/`tool_result` events, source
   endpoints, `/api/config`) is now validated and stable to build against.

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
