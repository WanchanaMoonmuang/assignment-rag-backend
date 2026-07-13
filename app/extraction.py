from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown


@dataclass
class ExtractedChunk:
    content: str
    chunk_type: str
    location: dict[str, Any]


class ExtractionError(ValueError):
    pass


def _label(location: dict[str, Any]) -> dict[str, Any]:
    location = dict(location)
    if "label" in location:
        return location
    start, end = location.get("start"), location.get("end")
    if location["type"] == "line":
        location["label"] = f"Lines {start}-{end}"
    elif location["type"] == "page":
        location["label"] = f"Page {start}" if start == end else f"Pages {start}-{end}"
    elif location["type"] == "row":
        location["label"] = f"Data rows {start}-{end}"
    return location


def _split_prose(
    text: str,
    location: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    chunk_type: str = "prose",
) -> list[ExtractedChunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        ExtractedChunk(part, chunk_type, _label(location))
        for part in splitter.split_text(text)
        if part.strip()
    ]


def _line_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[ExtractedChunk]:
    lines = text.splitlines() or [text]
    chunks: list[ExtractedChunk] = []
    start, buffer, size = 1, [], 0
    for number, line in enumerate(lines, start=1):
        addition = len(line) + (1 if buffer else 0)
        if buffer and size + addition > chunk_size:
            chunks.append(
                ExtractedChunk(
                    "\n".join(buffer),
                    "prose",
                    _label({"type": "line", "start": start, "end": number - 1}),
                )
            )
            buffer, size, start = [], 0, number
        if len(line) > chunk_size:
            if buffer:
                chunks.append(
                    ExtractedChunk(
                        "\n".join(buffer),
                        "prose",
                        _label({"type": "line", "start": start, "end": number - 1}),
                    )
                )
                buffer, size = [], 0
            chunks.extend(
                _split_prose(
                    line,
                    {"type": "line", "start": number, "end": number},
                    chunk_size,
                    chunk_overlap,
                )
            )
            start = number + 1
            continue
        buffer.append(line)
        size += addition
    if buffer:
        chunks.append(
            ExtractedChunk(
                "\n".join(buffer),
                "prose",
                _label({"type": "line", "start": start, "end": len(lines)}),
            )
        )
    return [chunk for chunk in chunks if chunk.content.strip()]


def extract_text(
    data: bytes, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[ExtractedChunk]:
    try:
        return _line_chunks(data.decode("utf-8-sig"), chunk_size, chunk_overlap)
    except UnicodeDecodeError as exc:
        raise ExtractionError("Text must be UTF-8") from exc


def _table_chunks(
    rows: list[list[str]], location: dict[str, Any], chunk_size: int
) -> list[ExtractedChunk]:
    if not rows:
        return []
    header = [str(cell or "") for cell in rows[0]]
    body = [[str(cell or "") for cell in row] for row in rows[1:]] or [[]]
    prefix = " | ".join(header) + "\n" + " | ".join("---" for _ in header)
    if len(prefix) + 1 >= chunk_size:
        raise ExtractionError("Table headers exceed configured chunk size")

    chunks: list[ExtractedChunk] = []
    batch: list[list[str]] = []

    def publish_batch() -> None:
        if batch:
            chunks.append(
                ExtractedChunk(
                    prefix + "\n" + "\n".join(" | ".join(row) for row in batch),
                    "table",
                    _label(location),
                )
            )

    for row in body:
        row_text = " | ".join(row)
        if len(prefix) + len(row_text) + 1 > chunk_size:
            publish_batch()
            batch.clear()
            parts = _split_prose(
                row_text,
                location,
                max(1, chunk_size - len(prefix) - 1),
                0,
                "table",
            )
            chunks.extend(
                ExtractedChunk(prefix + "\n" + part.content, "table", part.location)
                for part in parts
            )
        elif batch and (
            len(batch) == 20
            or len(prefix)
            + sum(len(" | ".join(item)) + 1 for item in batch)
            + len(row_text)
            + 1
            > chunk_size
        ):
            publish_batch()
            batch.clear()
            batch.append(row)
        else:
            batch.append(row)
    publish_batch()
    return chunks


def extract_pdf(
    data: bytes, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[ExtractedChunk]:
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ExtractionError("PDF could not be read") from exc
    if document.is_encrypted:
        document.close()
        raise ExtractionError("PDF is encrypted")
    chunks: list[ExtractedChunk] = []
    for page_number, page in enumerate(document, start=1):
        location = {"type": "page", "start": page_number, "end": page_number}
        text = page.get_text("text").strip()
        if text:
            chunks.extend(_split_prose(text, location, chunk_size, chunk_overlap))
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []
        for table_number, table in enumerate(tables, start=1):
            table_location = {
                **location,
                "table_index": table_number,
                "label": f"Page {page_number}, Table {table_number}",
            }
            chunks.extend(_table_chunks(table.extract(), table_location, chunk_size))
    document.close()
    if not chunks:
        raise ExtractionError("PDF does not contain extractable text")
    return chunks


def _markdown_chunks(
    markdown: str, chunk_size: int, chunk_overlap: int
) -> list[ExtractedChunk]:
    path: list[str] = []
    prose: list[str] = []
    table: list[str] = []
    chunks: list[ExtractedChunk] = []

    def location() -> dict[str, Any]:
        return {
            "type": "section",
            "label": " > ".join(path) or "Document",
            "section_path": list(path),
        }

    def flush() -> None:
        nonlocal prose, table
        if prose:
            chunks.extend(_split_prose("\n".join(prose), location(), chunk_size, chunk_overlap))
        if table:
            rows = [
                [cell.strip() for cell in line.strip().strip("|").split("|")]
                for line in table
                if "---" not in line
            ]
            chunks.extend(_table_chunks(rows, location(), chunk_size))
        prose, table = [], []

    for line in markdown.splitlines():
        if line.startswith("#") and line.lstrip("#").startswith(" "):
            flush()
            depth = len(line) - len(line.lstrip("#"))
            path[depth - 1 :] = [line[depth:].strip()]
        elif line.strip().startswith("|") and line.strip().endswith("|"):
            if prose:
                flush()
            table.append(line)
        else:
            if table:
                flush()
            prose.append(line)
    flush()
    return [chunk for chunk in chunks if chunk.content.strip()]


def extract_markdown(
    data: bytes,
    suffix: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[ExtractedChunk]:
    try:
        with NamedTemporaryFile(suffix=suffix) as temp:
            temp.write(data)
            temp.flush()
            markdown = MarkItDown().convert(Path(temp.name)).text_content
    except Exception as exc:
        raise ExtractionError("Document could not be converted") from exc
    chunks = _markdown_chunks(markdown, chunk_size, chunk_overlap)
    if not chunks:
        raise ExtractionError("Document does not contain extractable text")
    return chunks


def extract_csv(
    data: bytes, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[ExtractedChunk]:
    del chunk_overlap
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ExtractionError("CSV is not valid UTF-8 CSV") from exc
    if len(rows) < 2 or not rows[0] or any(not header.strip() for header in rows[0]):
        raise ExtractionError("CSV must include headers and data rows")
    headers = rows[0]
    if len(set(headers)) != len(headers) or any(len(row) != len(headers) for row in rows[1:]):
        raise ExtractionError("CSV rows must match the header columns")
    data_rows = rows[1:]
    numeric: list[str] = []
    for index, header in enumerate(headers):
        try:
            values = [float(row[index]) for row in data_rows if row[index].strip()]
        except ValueError:
            continue
        if values:
            numeric.append(
                f"{header}: min={min(values):g}, max={max(values):g}, mean={sum(values) / len(values):g}"
            )
    summary = f"Dataset: {len(data_rows)} rows, {len(headers)} columns.\nColumns: {', '.join(headers)}"
    if numeric:
        summary += "\nNumeric summary: " + "; ".join(numeric)
    chunks = _split_prose(summary, {"type": "dataset", "label": "Dataset summary"}, chunk_size, 0, "summary")
    prefix = " | ".join(headers) + "\n" + " | ".join("---" for _ in headers)
    if len(prefix) + 1 >= chunk_size:
        raise ExtractionError("CSV headers exceed configured chunk size")
    batch: list[list[str]] = []
    batch_start = 1

    def publish_batch() -> None:
        if batch:
            chunks.append(
                ExtractedChunk(
                    prefix + "\n" + "\n".join(" | ".join(row) for row in batch),
                    "table",
                    _label(
                        {
                            "type": "row",
                            "start": batch_start,
                            "end": batch_start + len(batch) - 1,
                        }
                    ),
                )
            )

    for row_number, row in enumerate(data_rows, start=1):
        row_text = " | ".join(row)
        if len(prefix) + len(row_text) + 1 > chunk_size:
            publish_batch()
            batch.clear()
            parts = _split_prose(
                row_text,
                {"type": "row", "start": row_number, "end": row_number},
                max(1, chunk_size - len(prefix) - 1),
                0,
                "table",
            )
            chunks.extend(
                ExtractedChunk(prefix + "\n" + part.content, "table", part.location)
                for part in parts
            )
            batch_start = row_number + 1
        elif batch and (len(batch) == 20 or len(prefix) + sum(len(" | ".join(item)) + 1 for item in batch) + len(row_text) + 1 > chunk_size):
            publish_batch()
            batch.clear()
            batch_start = row_number
            batch.append(row)
        else:
            batch.append(row)
    publish_batch()
    return chunks


def extract_json(
    data: bytes, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[ExtractedChunk]:
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionError("JSON is not valid UTF-8 JSON") from exc
    records = value if isinstance(value, list) else [value]
    chunks: list[ExtractedChunk] = []
    for index, record in enumerate(records, start=1):
        if isinstance(value, list):
            location = {
                "type": "record",
                "start": index,
                "end": index,
                "label": f"Record {index}",
                "path": f"$[{index - 1}]",
            }
        else:
            location = {"type": "path", "label": "Root object", "path": "$"}
        parts = _split_prose(
            json.dumps(record, ensure_ascii=False, indent=2),
            location,
            chunk_size,
            chunk_overlap,
            "json_record",
        )
        for part_number, chunk in enumerate(parts, start=1):
            if len(parts) > 1:
                chunk.location = {
                    **chunk.location,
                    "chunk_part": part_number,
                    "chunk_parts": len(parts),
                    "label": f"{chunk.location['label']}, part {part_number} of {len(parts)}",
                }
        chunks.extend(parts)
    return chunks


def extract(
    data: bytes, suffix: str, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[ExtractedChunk]:
    suffix = suffix.lower()
    if suffix == ".txt":
        return extract_text(data, chunk_size, chunk_overlap)
    if suffix == ".pdf":
        return extract_pdf(data, chunk_size, chunk_overlap)
    if suffix == ".docx":
        return extract_markdown(data, suffix, chunk_size, chunk_overlap)
    if suffix == ".csv":
        return extract_csv(data, chunk_size, chunk_overlap)
    if suffix == ".json":
        return extract_json(data, chunk_size, chunk_overlap)
    raise ExtractionError("Unsupported file type")
