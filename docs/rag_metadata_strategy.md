# RAG Metadata Strategy

## 1. Purpose

This document defines how Knowledge Assistant records document, chunk, and
user-supplied metadata for ingestion, retrieval, citations, and source
inspection. It covers pasted text and every supported file format: TXT,
PDF, DOCX, CSV, and JSON.

The schemas below are the Scope 3 target contract. Fields already produced by
the backend retain their current meaning; format-aware fields become available
when Scope 3 extraction is complete.

## 2. Principles

- Browser-provided MIME types are advisory. The backend validates the filename
  extension, file signature where applicable, decoding, and parser result.
- `source_format` is the backend-validated format and controls extraction.
- System metadata and user metadata remain separate. Users cannot overwrite
  system-owned fields.
- User metadata is stored on the document and copied to each chunk so retrieval
  results do not require a document lookup.
- Metadata is not prepended to embedded chunk content by default. This avoids
  changing semantic similarity with administrative values.
- Each chunk records enough source information to produce and reopen a citation
  without rerunning extraction or retrieval.
- Internal storage identifiers may be persisted but are not exposed as public
  object URLs.

## 3. Common Document Metadata

Every published document uses the following core fields:

```json
{
  "_id": "doc_01...",
  "document_name": "quarterly-report.pdf",
  "source_kind": "file",
  "source_format": "pdf",
  "content_type": "application/pdf",
  "byte_size": 1843200,
  "original_object_name": "documents/doc_01/original.pdf",
  "chunks_count": 24,
  "metadata": {
    "department": "finance",
    "fiscal_year": "2025"
  },
  "created_at": "2026-07-13T10:00:00Z",
  "updated_at": "2026-07-13T10:00:00Z"
}
```

| Field | Meaning |
| --- | --- |
| `_id` | Backend-generated document identifier |
| `document_name` | User-visible filename or pasted-text title |
| `source_kind` | `file` or `text` |
| `source_format` | Validated format: `txt`, `pdf`, `docx`, `csv`, or `json` |
| `content_type` | Backend-normalized MIME type |
| `byte_size` | Original input size in bytes |
| `original_object_name` | Private GCS object key, or `null` when not stored as an object |
| `chunks_count` | Number of published searchable chunks |
| `metadata` | User-supplied JSON-compatible key/value metadata |
| `created_at`, `updated_at` | Backend timestamps in UTC |

The browser-provided MIME type may be retained as `uploaded_content_type` for
diagnostics, but it never controls parsing.

## 4. Common Chunk Metadata

Every searchable chunk contains:

```json
{
  "_id": "chunk_01...",
  "document_id": "doc_01...",
  "document_name": "quarterly-report.pdf",
  "source_format": "pdf",
  "chunk_index": 4,
  "chunk_type": "prose",
  "location": {
    "type": "page",
    "start": 3,
    "end": 3,
    "label": "Page 3"
  },
  "metadata": {
    "department": "finance",
    "fiscal_year": "2025"
  }
}
```

`chunk_index` is a one-based sequence within the extracted document. It enables
stable ordering and surrounding-chunk lookup. `location.label` is display-ready;
the structured fields support navigation and citation inspection.

## 5. Supported Types

| Input | Normalized MIME type | Authoritative validation | Citation location |
| --- | --- | --- | --- |
| Pasted text | `text/plain; charset=utf-8` | UTF-8 encoding and size | Line range |
| `.txt` | `text/plain` | UTF-8 decoding | Line range |
| `.pdf` | `application/pdf` | PDF signature and parser | Page or page range |
| `.docx` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | ZIP package containing `word/document.xml` | Heading or section path |
| `.csv` | `text/csv` | UTF-8/BOM decoding and CSV parser | Data-row range |
| `.json` | `application/json` | UTF-8/BOM decoding and JSON parser | Record index or root path |

### 5.1 Pasted Text

Pasted text has no uploaded filename or trusted client MIME value.

```json
{
  "source_kind": "text",
  "source_format": "txt",
  "content_type": "text/plain; charset=utf-8",
  "byte_size": 4096,
  "original_object_name": null
}
```

Chunks use `chunk_type: "prose"` and exact normalized line ranges:

```json
{
  "chunk_type": "prose",
  "location": {
    "type": "line",
    "start": 1,
    "end": 18,
    "label": "Lines 1-18"
  }
}
```

The 20 MiB limit is measured from the UTF-8 encoded bytes before job creation.

### 5.2 TXT

Advisory MIME values commonly include `text/plain`,
`application/octet-stream`, or an empty value. Successful UTF-8 decoding and
the `.txt` extension are authoritative.

Document-specific metadata:

```json
{
  "source_format": "txt",
  "content_type": "text/plain",
  "encoding": "utf-8",
  "has_bom": false,
  "line_count": 286
}
```

Chunks use `prose` and line locations such as `Lines 42-58`. A UTF-8 BOM is
accepted and removed. Invalid UTF-8 or binary content fails ingestion.

### 5.3 PDF

Advisory MIME values commonly include `application/pdf` and
`application/octet-stream`. The backend requires a PDF signature and successful
page-aware parsing.

Document-specific metadata:

```json
{
  "source_format": "pdf",
  "content_type": "application/pdf",
  "page_count": 16,
  "encrypted": false
}
```

Prose chunks preserve page boundaries:

```json
{
  "chunk_type": "prose",
  "location": {
    "type": "page",
    "start": 4,
    "end": 4,
    "label": "Page 4"
  }
}
```

Tables use `chunk_type: "table"` and include `table_index` when the extractor
can identify separate tables:

```json
{
  "chunk_type": "table",
  "location": {
    "type": "page",
    "start": 7,
    "end": 7,
    "label": "Page 7, Table 1",
    "table_index": 1
  }
}
```

Encrypted, malformed, unreadable, and image-only PDFs fail with safe actionable
errors. OCR is outside the POC scope.

### 5.4 DOCX

The normalized MIME type is
`application/vnd.openxmlformats-officedocument.wordprocessingml.document`.
`application/octet-stream` may be received from a browser, but the backend
validates the ZIP package and requires `word/document.xml`.

Document-specific metadata:

```json
{
  "source_format": "docx",
  "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "section_count": 12
}
```

Prose, headings, and lists use a section path:

```json
{
  "chunk_type": "prose",
  "location": {
    "type": "section",
    "label": "Operations > Incident Response",
    "section_path": ["Operations", "Incident Response"]
  }
}
```

Tables are extracted as separate `chunk_type: "table"` chunks (rendered as
pipe-delimited Markdown) and reuse the containing section path as their
location — they do not carry a table index (only PDF tables do). DOCX citations
do not use page numbers because pagination depends on the rendering engine,
fonts, and page settings.

### 5.5 CSV

Advisory MIME values may include `text/csv`, `application/csv`,
`application/vnd.ms-excel`, and `application/octet-stream`. The extension,
UTF-8/BOM decoding, and CSV parser determine validity.

Document-specific metadata:

```json
{
  "source_format": "csv",
  "content_type": "text/csv",
  "encoding": "utf-8",
  "row_count": 480,
  "column_count": 7,
  "columns": ["date", "region", "product", "quantity", "unit_price", "revenue", "status"]
}
```

Row batches preserve headers and use locations such as:

```json
{
  "chunk_type": "table",
  "location": {
    "type": "row",
    "start": 21,
    "end": 40,
    "label": "Data rows 21-40"
  }
}
```

A dataset summary may be published as `chunk_type: "summary"` with dimensions
and useful numeric aggregates such as minimum, maximum, and mean. These values
belong in searchable extracted content unless a later filtering requirement
needs them as database metadata.

Empty CSV files, missing usable headers, invalid decoding, and structures that
cannot be parsed reliably fail ingestion explicitly.

### 5.6 JSON

Advisory MIME values commonly include `application/json`, `text/json`, and
`application/octet-stream`. The standard JSON parser determines validity.

A root array records its structure:

```json
{
  "source_format": "json",
  "content_type": "application/json",
  "root_type": "array",
  "record_count": 125
}
```

Each list entry receives a one-based display index and a zero-based JSON path:

```json
{
  "chunk_type": "json_record",
  "location": {
    "type": "record",
    "start": 17,
    "end": 17,
    "label": "Record 17",
    "path": "$[16]"
  }
}
```

A root object uses `root_type: "object"`, `record_count: 1`, and location path
`$` with label `Root object`. If one record exceeds the chunk limit, each part
retains the same record or root path and adds `chunk_part` and `chunk_parts`.

Invalid JSON, trailing data, and invalid UTF-8 fail before embedding.

## 6. User Metadata

The frontend accepts a generic JSON-compatible key/value object, for example:

```json
{
  "department": "legal",
  "category": "contract",
  "effective_year": 2026,
  "confidential": true
}
```

The object is stored on the document and copied unchanged to every chunk. The
current POC returns it with retrieval context but does not use it as a ranking
factor or retrieval filter.

The following system fields are reserved and cannot be overwritten through
user metadata:

```text
_id
document_id
document_name
source_kind
source_format
content_type
original_object_name
chunk_index
chunk_type
location
embedding
created_at
updated_at
```

## 7. Retrieval and Citation Use

Hybrid retrieval ranks extracted chunk content through lexical and vector
search. Metadata accompanies each result and supplies:

- document and chunk identity;
- filename and validated source format;
- chunk type and sequence;
- format-aware location;
- retrieval score;
- user metadata needed by the client or future filters.

Persisted assistant messages retain their complete source list. Reopening a
conversation therefore reproduces the original citations without running
retrieval again.

## 8. Security and API Boundaries

- GCS buckets and objects remain private.
- The frontend receives authenticated source-content responses, not public GCS
  URLs or cloud credentials.
- Internal object names are exposed only where an authenticated backend workflow
  requires them.
- Credentials, authorization headers, prompts, document text, and embeddings are
  never included in operational logs.
- Extracted content is rendered as escaped text or sanitized Markdown; citation
  highlighting uses semantic `mark` elements rather than unsanitized HTML.

## 9. Current Limitations

- Metadata does not currently filter or boost retrieval.
- PDF OCR is not included, so image-only documents fail ingestion.
- DOCX locations are section-based rather than page-based.
- CSV summaries are intentionally basic and are not a statistical analysis
  engine.
- Browser MIME values vary by platform; parser validation remains authoritative.
- Format-specific metadata and locations described here depend on completion of
  Scope 3 extraction and publication.
