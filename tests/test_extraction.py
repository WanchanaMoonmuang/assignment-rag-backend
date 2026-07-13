from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import fitz
import pytest

from app.extraction import ExtractionError, extract_csv, extract_json, extract_markdown, extract_pdf, extract_text


def test_text_extraction_uses_exact_line_ranges() -> None:
    chunks = extract_text(b"one\ntwo\nthree\nfour", chunk_size=8, chunk_overlap=1)

    assert [chunk.location["label"] for chunk in chunks] == ["Lines 1-2", "Lines 3-3", "Lines 4-4"]


def test_pdf_extraction_preserves_page_location() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "The refund period is thirty days.")
    chunks = extract_pdf(document.tobytes())

    assert chunks[0].chunk_type == "prose"
    assert chunks[0].location["label"] == "Page 1"


def test_csv_extraction_adds_summary_and_bounded_row_locations() -> None:
    data = "name,value\n" + "\n".join(f"item-{index},{index}" for index in range(1, 22))
    chunks = extract_csv(data.encode())

    assert chunks[0].chunk_type == "summary"
    assert "mean=11" in chunks[0].content
    assert [chunk.location["label"] for chunk in chunks[1:]] == [
        "Data rows 1-20",
        "Data rows 21-21",
    ]


def test_csv_extraction_rejects_inconsistent_rows() -> None:
    with pytest.raises(ExtractionError, match="header columns"):
        extract_csv(b"name,value\nfirst,1,extra")


def test_json_extraction_preserves_record_path_when_split() -> None:
    chunks = extract_json(b'[{"name": "first"}, {"name": "second"}]', chunk_size=20, chunk_overlap=1)

    assert chunks[0].location["path"] == "$[0]"
    assert chunks[-1].location["path"] == "$[1]"


def test_json_extraction_rejects_invalid_json() -> None:
    with pytest.raises(ExtractionError, match="valid UTF-8 JSON"):
        extract_json(b"{invalid}")


def test_docx_markdown_extraction_tracks_section_and_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Converter:
        def convert(self, path: Any) -> Any:
            return SimpleNamespace(
                text_content="# Operations\nEscalate incidents.\n| Code | Meaning |\n| --- | --- |\n| ESC-42 | Priority |"
            )

    monkeypatch.setattr("app.extraction.MarkItDown", Converter)
    chunks = extract_markdown(b"docx bytes", ".docx")

    assert {chunk.chunk_type for chunk in chunks} == {"prose", "table"}
    assert all(chunk.location["label"] == "Operations" for chunk in chunks)



def test_csv_extraction_respects_chunk_size_for_long_rows() -> None:
    chunks = extract_csv(b"name,value\nitem," + b"x" * 100, chunk_size=30, chunk_overlap=0)

    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    assert len(table_chunks) > 1
    assert all(chunk.location["label"] == "Data rows 1-1" for chunk in table_chunks)
    assert all(chunk.content.startswith("name | value") for chunk in table_chunks)



def test_table_chunks_respect_chunk_size_for_long_rows() -> None:
    from app.extraction import _table_chunks

    chunks = _table_chunks(
        [["name", "value"], ["item", "x" * 100]],
        {"type": "section", "label": "Operations"},
        30,
    )

    assert len(chunks) > 1
    assert all(chunk.content.startswith("name | value") for chunk in chunks)
    assert all(chunk.location["label"] == "Operations" for chunk in chunks)
