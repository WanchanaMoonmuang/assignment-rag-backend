# RAG System Data Flow

The technical mechanics, split into the two phases that share MongoDB. **Ingestion** runs
offline in the worker and turns a document into searchable, embedded chunks. **Query** runs
online in the API: it retrieves the most relevant chunks with hybrid search, builds a
prompt, and streams a cited answer from Gemini.

Ingestion is **not** a single universal pipeline — each file extension is dispatched to
its own extractor (`app/extraction.py:extract()`). Only `.docx` is converted with
MarkItDown; `.pdf` is parsed directly with fitz/PyMuPDF (never MarkItDown, to keep page
boundaries and avoid re-emitting table text as prose), and `.txt`/`.csv`/`.json` use their
own custom chunkers with no conversion step at all.

```mermaid
flowchart TD
    subgraph Ingest["📥 Ingestion — offline worker"]
        direction TB
        I1["Upload file / paste text"] --> I2["API: validate<br/>(type, size ≤ 20 MiB)"]
        I2 --> I3["Store original in private GCS"]
        I2 --> I4["Create ingestion job<br/>(queued) in MongoDB"]
        I4 --> I5["Worker claims job with a lease<br/>(bounded retries)"]
        I5 --> I5t[".txt: line-window chunker<br/>no conversion; line-range location"]
        I5 --> I5p[".pdf: fitz/PyMuPDF directly<br/>find table bboxes first, then prose<br/>via get_text(blocks) EXCLUDING those<br/>bboxes — prevents table/prose duplication;<br/>page-aware location"]
        I5 --> I5d[".docx: MarkItDown → Markdown<br/>(the only format that uses MarkItDown)<br/>then heading-path chunker;<br/>section-path location"]
        I5 --> I5c[".csv: row-batch chunker<br/>+ one synthesized 'Dataset summary' chunk<br/>(counts, columns, numeric min/max/mean);<br/>row-range location"]
        I5 --> I5j[".json: record-aware chunker<br/>one chunk per array element / root object;<br/>record-index or JSON-path location"]
        I5t --> I8["Embed each chunk<br/>Gemini embeddings, 768-dim"]
        I5p --> I8
        I5d --> I8
        I5c --> I8
        I5j --> I8
        I8 --> I9["Write document + chunks + vectors"]
        I9 --> I10["Publish: mark completed<br/>→ searchable"]
    end

    subgraph Query["🔎 Query — online API"]
        direction TB
        Q1["Question + Top K + recent history"] --> Q2{Top K = 0?}
        Q2 -->|Yes| Q6["Skip retrieval<br/>(foundation knowledge only)"]
        Q2 -->|No| Q3["Embed the query<br/>(Gemini embeddings)"]
        Q3 --> Q4["Hybrid retrieve:<br/>Atlas Vector Search ∥ Atlas BM25"]
        Q4 --> Q5["Fuse with $rankFusion<br/>→ take Top K chunks"]
        Q5 --> Q7
        Q6 --> Q7["Build prompt: history + retrieved<br/>context + question, trimmed to<br/>token budget"]
        Q7 --> Q8["Gemini generates<br/>(optional calculator tool loop)"]
        Q8 --> Q9["Stream tokens + [N] citations<br/>+ tool activity over SSE"]
        Q9 --> Q10["Persist assistant message:<br/>answer + sources + tool activity"]
        Q10 --> Q11["Render in UI: answer,<br/>citation drawer, tool activity"]
    end

    DB[("🗄️ MongoDB Atlas<br/>documents · chunks + vectors ·<br/>conversations · ingestion jobs")]
    GCS[("🔒 Google Cloud Storage<br/>private original files")]

    I4 -.-> DB
    I9 -.-> DB
    I10 -.-> DB
    I3 -.-> GCS
    Q4 -.->|reads chunks + vectors| DB
    Q10 -.-> DB
    Q11 -.->|"PDF preview reads the original<br/>at the cited page"| GCS

    classDef store fill:#f5f5f5,stroke:#666,color:#222;
    classDef markitdown fill:#fff3e0,stroke:#e65100,color:#222;
    class DB,GCS store;
    class I5d markitdown;
```
