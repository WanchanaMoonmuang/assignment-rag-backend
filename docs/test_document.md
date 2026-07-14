# Backend Local Acceptance Test

Run from `poc-rag-backend` with the local server on port `8080`:

```bash
env -u MONGODB_URI uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

API docs:

```text
http://127.0.0.1:8080/api/docs
http://172.25.0.4:8080/api/docs
```

## Acceptance Criteria

- `GET /api/health` returns `{"status":"ok"}`.
- `GET /api/docs` loads Swagger UI.
- Protected endpoints reject missing bearer auth.
- `POST /api/auth/login` rejects invalid credentials.
- `POST /api/auth/login` returns a bearer token for valid credentials.
- `GET /api/auth/me` returns the configured admin username.
- `POST /api/ingest` ingests text and returns `document_id` plus `chunks_created`.
- `GET /api/documents` lists the ingested document.
- `POST /api/chat` creates a conversation and returns `answer` plus `sources`.
- `POST /api/chat/stream` emits `conversation`, `token`, `sources`, and `done` events.
- `GET /api/conversations` and `GET /api/conversations/{conversation_id}` return stored chat history.
- `DELETE /api/conversations/{conversation_id}` deletes the test conversation.
- `DELETE /api/documents/{document_id}` deletes the test document and chunks.

## Latest Local Result

Full automated smoke test passed:

```text
22 / 22 passed
```

Real mock RAG scenario passed against the local server on port `8080` using real
MongoDB Atlas and Gemini services:

```text
health 200 ok
protected_without_auth 401
login 200
me 200 admin
ingest 200
document_id doc_d10fc6aad8f24ba1a64b7613cc74665b chunks 1
document_listed True
chat_attempt 1 200, answer_has_code False, sources 0
chat_attempt 2 200, answer_has_code True, sources 1
result PASS
cleanup_conversation 200
cleanup_document 200
```

## Vertex AI Local Result

Vertex AI mode was tested against the local server on port `8080` using real
MongoDB Atlas, the service account key at
`/home/developer/.config/gcloud/gcp_key.json`, and `GOOGLE_CLOUD_LOCATION=us`.
The multi-region `us` location is required for the configured Vertex model names;
`us-central1` and `asia-southeast1` returned `404 NOT_FOUND` for
`gemini-embedding-2` after IAM permission was granted.

Server command used:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/home/developer/.config/gcloud/gcp_key.json \
GEMINI_PROVIDER=vertex_ai \
GOOGLE_CLOUD_PROJECT=$GCP_PROJECT_ID \
GOOGLE_CLOUD_LOCATION=us \
uv run uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Direct Vertex SDK smoke test:

```text
provider=vertex_ai
project=my-test-project-425807
location=us
embed=ok count=1 dims=768
generate=ok
```

Curl functional test result with timings (`time_starttransfer`, `time_total`):

```text
health 200 0.001973 0.002078
protected_without_auth 401 0.003653 0.003788
login 200 0.004221 0.004373
me 200 0.002745 0.002832
ingest 200 2.343382 2.343517
chat_attempt_1 200 1.720237 1.720573
chat_attempt_1_result sources=3 answer_has_code=false conversation_id_prefix=conv_c26
followup_chat 200 1.588428 1.588633
followup_result sources=3 answer_has_refund_window=false
history 200 0.038149 0.038285
history_result messages=4
cleanup_conversation 200 0.040904 0.041126
cleanup_document 200 0.143377 0.143617
```

Functional status:

- PASS: local backend starts in Vertex AI mode.
- PASS: auth, protected endpoint rejection, ingestion, retrieval, citations, chat history storage, and cleanup all returned expected HTTP status codes.
- PASS: first chat and follow-up both returned `sources=3`.
- PASS: selected conversation history returned `4` messages after two non-streaming chat turns.
- NOTE: answer text did not include the exact escalation code or refund window even though sources were returned; this is an answer-quality/prompt behavior to review separately from API functionality.

## Notes

MongoDB Atlas Vector Search is eventually consistent. A newly ingested document
may not be immediately retrievable; retry after a short delay before treating
retrieval as failed.

Do not commit `.env` or real credentials.

## Frontend Integration Flow

Use this request order for frontend integration:

1. `GET /api/health`
2. `POST /api/auth/login`
3. Store `access_token`.
4. Send `Authorization: Bearer <access_token>` on protected API calls.
5. `POST /api/ingest` to upload plain text.
6. `GET /api/documents` to refresh the document list.
7. Start a new conversation with `POST /api/chat` or `POST /api/chat/stream` without `conversation_id`.
8. Continue an existing conversation by passing the returned `conversation_id`.
9. Use `GET /api/conversations` for the conversation list.
10. Use `GET /api/conversations/{conversation_id}` for message history.
11. Delete test data with `DELETE /api/conversations/{conversation_id}` and `DELETE /api/documents/{document_id}`.

Response shape notes:

- Login returns `access_token` and `token_type`.
- Ingest returns `document_id`, `document_name`, and `chunks_created`.
- Document list returns `{ "documents": [...] }`.
- Chat returns `conversation_id`, `answer`, and `sources`.
- Streaming chat uses server-sent events with `conversation`, `token`, `sources`, and `done` events.

## Curl Smoke Commands

Run these while the backend server is running on port `8080`.

```bash
BASE=http://127.0.0.1:8080/api
```

Health:

```bash
curl "$BASE/health"
```

Protected endpoint without token should return `401`:

```bash
curl -i "$BASE/documents"
```

Login and capture token. Use the same password as `AUTH_PASSWORD` in local `.env`:

```bash
read -rsp "AUTH_PASSWORD: " AUTH_PASSWORD
echo
TOKEN=$(curl -s -X POST "$BASE/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"admin\",\"password\":\"$AUTH_PASSWORD\"}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
```

Current user:

```bash
curl "$BASE/auth/me" \
  -H "Authorization: Bearer $TOKEN"
```

Ingest a test document and capture `document_id`:

```bash
DOC_ID=$(curl -s -X POST "$BASE/ingest" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "document_name": "curl-functional-test.txt",
    "content": "ALPHA-7429 is the customer escalation code for Bangkok onboarding failures. Refunds are accepted within 30 days.",
    "metadata": {"source": "curl_functional_test"}
  }' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["document_id"])')
echo "$DOC_ID"
```

List documents:

```bash
curl "$BASE/documents" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -m json.tool
```

Start a new non-streaming conversation by omitting `conversation_id`:

```bash
CHAT_RESPONSE=$(curl -s -X POST "$BASE/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the escalation code for Bangkok onboarding failures?","top_k":5}')
echo "$CHAT_RESPONSE" | python3 -m json.tool
CONVERSATION_ID=$(echo "$CHAT_RESPONSE" | python3 -c 'import sys,json; print(json.load(sys.stdin)["conversation_id"])')
```

If `sources` is empty, wait 10-20 seconds and retry the same question because
Atlas Vector Search indexing is asynchronous.

Continue the same conversation by sending `conversation_id`:

```bash
curl -X POST "$BASE/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"$CONVERSATION_ID\",\"question\":\"What is the refund window?\",\"top_k\":5}" \
  | python3 -m json.tool
```

Start a new streaming conversation by omitting `conversation_id`:

```bash
curl -N -X POST "$BASE/chat/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund window?","top_k":5}'
```

Continue an existing streaming conversation:

```bash
curl -N -X POST "$BASE/chat/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"$CONVERSATION_ID\",\"question\":\"Repeat the escalation code.\",\"top_k\":5}"
```

List conversations:

```bash
curl "$BASE/conversations" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -m json.tool
```

Get one conversation history:

```bash
curl "$BASE/conversations/$CONVERSATION_ID" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -m json.tool
```

Cleanup by IDs from responses:

```bash
curl -X DELETE "$BASE/conversations/$CONVERSATION_ID" \
  -H "Authorization: Bearer $TOKEN"
```

```bash
curl -X DELETE "$BASE/documents/$DOC_ID" \
  -H "Authorization: Bearer $TOKEN"
```

## V2 Scope 1 and 2 Local Curl Result

Run date: 2026-07-13  
Backend process: `uvicorn app.main:app --host 0.0.0.0 --port 8080`  
Worker process: `python -m app.worker`  
Provider: Vertex AI, location `us`  
Verified stacked commits: `55f96ff` generation and `1531539` durable ingestion.

The configured project MongoDB collections were cleared before this scenario:

```text
documents: 4 deleted
document_chunks: 5 deleted
conversations: 2 deleted
ingestion_jobs: 0 deleted
```

Authenticated curl scenario results:

```text
health=200 0.001028s
unauthenticated documents=401 0.000906s
authenticated current user=200 0.002748s
POST /api/ingestions/text -> queued job
worker terminal status=completed
POST /api/chat top_k=0 -> sources=0
POST /api/chat top_k=5 -> sources=1
POST /api/chat/stream top_k=0 -> done event=1
invalid top_k=21 -> 422
mismatched PDF content -> 415
deprecated POST /api/ingest -> queued
DELETE test document -> success
DELETE test conversation -> success
```

Functional status:

- PASS: Vertex AI backend starts locally.
- PASS: authentication, bearer rejection, and current-user API.
- PASS: durable text ingestion, job polling, worker publication, and document cleanup.
- PASS: Top K zero bypasses retrieval and returns no citations.
- PASS: retrieved chat returns a source after Atlas indexing.
- PASS: SSE returns a `done` event for no-retrieval chat.
- PASS: request validation rejects Top K outside `0-20`.
- PASS: file content that does not match a PDF signature is rejected before GCS upload/job creation.
- PASS: the deprecated V1 ingestion endpoint now queues the V2 text-ingestion job.

Not yet included in this run:

- PDF, DOCX, CSV, and JSON conversion/extraction. Scope 3 is still in progress.
- GCS successful-file upload and cleanup validation. A bucket named `poc-rag-58` is available and will be exercised once file extraction is connected.


## 2026-07-13: Pre-merge Backend V2 Curl Validation

Target: `feature/backend-v2-observability` (includes extraction, hybrid retrieval,
generation, citations/tools, and observability). API: `http://127.0.0.1:8081`.

Use the configured credentials without printing the bearer token:

```bash
set -a; source .env; set +a
USER=${AUTH_USERNAME:-admin}
TOKEN=$(curl -sS -X POST http://127.0.0.1:8081/api/auth/login \
  -H "Content-Type: application/json" \
  --data "{\"username\":\"$USER\",\"password\":\"$AUTH_PASSWORD\"}" \
  | sed -n 's/.*\"access_token\":\"\([^\"]*\)\".*/\1/p')
```

| Scenario | Curl endpoint | Result | Latency |
| --- | --- | --- | --- |
| Health | `GET /api/health` | PASS: `200 {"status":"ok"}` | not captured |
| Login | `POST /api/auth/login` | PASS: `200`; `.env` has a blank `AUTH_USERNAME`, so use `admin` fallback | 9 ms |
| Runtime config | `GET /api/config` | PASS: `200`; Top K default `5`, range `0-20`, upload limit `20971520`, supported extensions returned | 6 ms |
| Text ingestion | `POST /api/ingestions/text` then poll `GET /api/ingestions/{job_id}` | BLOCKED: enqueue returned `202` in 1.03 s; worker marked the job `failed` with safe `ingestion_failed` | n/a |
| Non-stream chat | `POST /api/chat` with `top_k: 0` | BLOCKED: API returned `500 Internal Server Error` in 374 ms | 374 ms |
| Direct Vertex basic generation | local `GeminiClient.generate` | PASS: returned a response | n/a |
| Direct Vertex calculator generation | local `GeminiClient.generate_with_calculator` | PASS: returned a response and provider usage `{input_tokens: 102, output_tokens: 9, total_tokens: 111}` | n/a |

Not executed because ingestion and API chat are blocked: file ingestion completion, hybrid retrieval answer/sources, citation chunk lookup, SSE chat events, and calculator SSE events.

### Pre-merge blockers

1. Capture the current API/worker traceback for the `POST /api/chat` 500 and worker
   `ingestion_failed`; both hide their internal error by design in the API response.
2. `uv run python -m app.check_config` incorrectly requires `GEMINI_API_KEY` when
   `GEMINI_PROVIDER=vertex_ai`; direct Vertex generation proves the configured ADC
   credentials and model access are working.

Do not merge the remaining V2 backend branches as validated until these two failures
are resolved and this curl matrix is rerun.
