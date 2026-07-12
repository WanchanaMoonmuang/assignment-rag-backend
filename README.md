# RAG Chatbot Backend

FastAPI backend for the RAG chatbot demo. API docs run at `/api/docs`.

## Local Setup

```bash
uv sync --extra dev
cp .env.example .env
uv run uvicorn app.main:app --reload
```

Required real-service values in `.env`:

For the PRD default demo account, set `AUTH_USERNAME=admin` and `AUTH_PASSWORD=adminRAG123`.

- `GEMINI_PROVIDER` (`developer_api` by default, or `vertex_ai`)
- `GEMINI_API_KEY` when `GEMINI_PROVIDER=developer_api`
- `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` when `GEMINI_PROVIDER=vertex_ai` (`GCP_PROJECT_ID` is accepted as a project fallback)
- `MONGODB_URI`
- `MONGODB_DATABASE`
- `MONGODB_VECTOR_INDEX`
- `JWT_SECRET_KEY`
- `AUTH_PASSWORD`

For Vertex AI, use application default credentials locally or the Cloud Run service account in deployment.

Optional tuning values:

- `RAG_TOP_K` default `5`
- `HISTORY_CONTEXT_WINDOW` default `8`
- `RAG_CHUNK_SIZE` default `900`
- `RAG_CHUNK_OVERLAP` default `150`

## Checks

```bash
uv run pytest
uv run ruff check .
uv run python -m app.check_config
```

`python -m app.check_config` pings MongoDB and checks that the configured Atlas
Vector Search index exists on the chunks collection.

## API

- `GET /api/health`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/ingest`
- `GET /api/documents`
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

## MongoDB Vector Search

Create an Atlas Vector Search index on `MONGODB_CHUNK_COLLECTION` using
`docs/mongodb-vector-index.json`. The vector dimensions must match
`GEMINI_EMBEDDING_DIMENSIONS` (`768` by default).
