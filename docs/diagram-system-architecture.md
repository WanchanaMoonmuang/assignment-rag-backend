# System Architecture

The two independently deployed applications, the two backend processes that share
MongoDB, and the external services they call. Tech stack labels are pulled directly from
`package.json` and `pyproject.toml`.

```mermaid
flowchart LR
    subgraph FE["🖥️ Frontend — assignment-rag-frontend"]
        direction TB
        Browser["React 19 + MUI v9 (Emotion)<br/>Vite build, TypeScript<br/>TanStack Query for server state<br/>eventsource-parser for SSE chat"]
    end

    subgraph BE["⚙️ Backend — assignment-rag-backend"]
        direction TB
        API["API process<br/>FastAPI + Uvicorn (async)<br/>JWT auth · hybrid retrieval<br/>Gemini calls · SSE streaming"]
        Worker["Worker process<br/>polls ingestion_jobs<br/>lease + bounded retries<br/>fitz/MarkItDown/custom extractors<br/>chunk + embed + publish"]
    end

    Mongo[("🗄️ MongoDB Atlas<br/>documents · chunks + vectors ·<br/>conversations · ingestion_jobs<br/>$vectorSearch + Atlas $search,<br/>fused via $rankFusion")]
    Gemini["🤖 Gemini<br/>(Developer API or Vertex AI)<br/>generation + embeddings<br/>+ calculator tool loop"]
    GCS[("🔒 Google Cloud Storage<br/>private original files")]

    Browser -->|"HTTPS/JSON + bearer token;<br/>raw fetch + SSE for /chat/stream"| API
    API -->|"answers, sources, job status"| Browser
    API <-->|"documents · chunks · vectors ·<br/>conversations"| Mongo
    API -->|"embed query · generate answer"| Gemini
    API -->|"enqueue ingestion job"| Mongo
    Worker -->|"claim job (lease)"| Mongo
    Worker -->|"embed chunks"| Gemini
    Worker -->|"write chunks + vectors,<br/>publish document"| Mongo
    Worker -->|"store original file"| GCS
    API -->|"stream original file<br/>for citation preview"| GCS

    classDef store fill:#f5f5f5,stroke:#666,color:#222;
    class Mongo,GCS store;
```

- **Two processes, one database.** The API and the worker are separate OS processes
  that never call each other directly — they coordinate only through MongoDB
  (`ingestion_jobs` as the queue). The Docker image runs both in one container for the
  POC; a real deployment would run them as separate services.
- **MarkItDown is not on the request path for every format** — only the worker's `.docx`
  branch uses it (see `diagram-rag-data-flow.md`). `.pdf` uses fitz/PyMuPDF directly.
- **Gemini is called from both processes**: the worker for embeddings during ingestion,
  the API for query embeddings, answer generation, and the calculator tool loop.
