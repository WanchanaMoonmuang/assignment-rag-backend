# RAG Chatbot Backend

FastAPI backend for the RAG chatbot demo. API docs run at `/api/docs`.

## Local Setup

```bash
uv sync --extra dev
cp .env.example .env
uv run uvicorn app.main:app --reload
uv run python -m app.worker
```

Required real-service values in `.env`:

For the PRD default demo account, set `AUTH_USERNAME=admin` and `AUTH_PASSWORD=adminRAG123`.

- `GEMINI_PROVIDER` (`developer_api` by default, or `vertex_ai`)
- `GEMINI_API_KEY` when `GEMINI_PROVIDER=developer_api`
- `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` when `GEMINI_PROVIDER=vertex_ai` (`GCP_PROJECT_ID` is accepted as a project fallback)
- `MONGODB_URI`
- `MONGODB_DATABASE`
- `MONGODB_VECTOR_INDEX`
- `MONGODB_SEARCH_INDEX` (defaults to `document_chunks_text_index`)
- `GCS_BUCKET_NAME` for private original files used by V2 file ingestion
- `JWT_SECRET_KEY`
- `AUTH_PASSWORD`

For Vertex AI, use application default credentials locally or the Cloud Run service account in deployment.

## Cloud Run POC Deployment

The image starts the API and ingestion worker in one container. Configure the
Cloud Run service with always-allocated CPU, minimum instances `1`, and maximum
instances `1`. Otherwise the worker can stop when there are no HTTP requests or
multiple instances can needlessly poll the same job queue. Set `BACKEND_CORS_ORIGINS`
to the deployed frontend origin, and grant the Cloud Run service account access to
Vertex AI and the private GCS bucket.

This is a POC deployment topology. Run the API and worker as separate services
for production workloads.

Optional tuning values:

- `RAG_TOP_K` default `5`
- `HISTORY_CONTEXT_WINDOW` default `8`
- `GEMINI_TEMPERATURE` default `0.2`
- `GENERATION_CONTEXT_TOKEN_BUDGET` default `32000` (a conservative UTF-8 byte upper bound; retrieved chunks are removed before oldest history)
- `RAG_CHUNK_SIZE` default `900`
- `RAG_CHUNK_OVERLAP` default `150`
- `INGESTION_JOB_LEASE_SECONDS` default `300`, `INGESTION_JOB_MAX_ATTEMPTS` default `3`, and `INGESTION_PROCESSING_TIMEOUT_SECONDS` default `600`

## Checks

```bash
uv run pytest
uv run ruff check .
uv run python -m app.check_config
```

`python -m app.check_config` pings MongoDB and checks that the configured Atlas
Vector Search and Atlas Search indexes exist on the chunks collection.

## API

- `GET /api/health`
- `GET /api/config`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/ingest` (deprecated; enqueues the same V2 text-ingestion job)
- `POST /api/ingestions/text`
- `POST /api/ingestions/file`
- `GET /api/ingestions/{job_id}`
- `GET /api/documents`
- `GET /api/documents/{document_id}/chunks/{chunk_id}`
- `GET /api/documents/{document_id}/file`
- `DELETE /api/documents/{document_id}`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}`
- `DELETE /api/conversations/{conversation_id}`
- `POST /api/chat`
- `POST /api/chat/stream`

Protected endpoints require:

```text
Authorization: Bearer <access_token>
```

## MongoDB Search Indexes

MongoDB Atlas 8.0 or later is required for hybrid retrieval.

Create an Atlas Vector Search index on `MONGODB_CHUNK_COLLECTION` using
`docs/mongodb-vector-index.json`. The vector dimensions must match
`GEMINI_EMBEDDING_DIMENSIONS` (`768` by default).

Create an Atlas Search index named by `MONGODB_SEARCH_INDEX` using
`docs/mongodb-search-index.json`. Retrieval combines equal-weight lexical and
vector result sets with MongoDB `$rankFusion`, then applies the requested Top K.
