"""Self-check for write_report()'s Markdown formatting.

Run with:
    uv run --extra eval python -m pytest evals/test_report.py
or directly:
    uv run --extra eval python evals/test_report.py

Not under tests/ (pyproject.toml's testpaths = ["tests"]) since it needs pandas, which
is only installed with the eval extra.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd

from app.settings import Settings
from evals.run_eval import write_report


class _FakeResult:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame


def _sample_rows() -> list[dict]:
    return [
        {
            "filename": "doc_a.pdf",
            "question": "What is the total?",
            "ground_truth": "The total is 42.",
            "answer": "The total is 42.",
            "sources": [
                {"document_name": "doc_a.pdf", "location": "Page 1", "score": 0.91},
                {"document_name": "doc_b.pdf", "location": "Page 3", "score": 0.40},
            ],
            "source_hit_rate": 0.5,
        },
        {
            "filename": "doc_a.pdf",
            "question": "Question with no retrievable answer",
            "ground_truth": "N/A",
            "answer": "",  # simulates a blank generation the fallback substitution missed
            "sources": [],
            "source_hit_rate": 0.0,
        },
    ]


def test_write_report_formats_metrics_and_citations() -> None:
    frame = pd.DataFrame(
        {
            "user_input": ["Q1", "Q2"],
            "retrieved_contexts": [["c1"], []],
            "response": ["A1", ""],
            "reference": ["R1", "R2"],
            "faithfulness": [1.0, math.nan],
            "llm_context_precision_with_reference": [0.8, 0.0],
            "context_recall": [0.6, 0.0],
        }
    )
    settings = Settings(gemini_api_key="test", mongodb_uri="mongodb://test")

    with tempfile.TemporaryDirectory() as tmp:
        write_report(_sample_rows(), _FakeResult(frame), Path(tmp), settings)
        text = next(Path(tmp).glob("*.md")).read_text(encoding="utf-8")

    # Human-readable metric names, not raw RAGAS column names.
    assert "Faithfulness" in text
    assert "llm_context_precision_with_reference" not in text
    assert "Context Precision" in text

    # NaN renders as N/A everywhere, never a raw "nan".
    assert "N/A" in text
    assert "nan" not in text.lower()

    # Citations: hit/miss marks against the expected source file.
    assert "✓ doc_a.pdf" in text
    assert "✗ doc_b.pdf" in text

    # Blank answers render as a visible placeholder, and answer text is blockquoted.
    assert "_(no answer generated)_" in text
    assert "> The total is 42." in text

    # Summary table header and separator rows have the same column count.
    lines = text.splitlines()
    header_line = next(line for line in lines if line.startswith("| # |"))
    separator_line = lines[lines.index(header_line) + 1]
    assert header_line.count("|") == separator_line.count("|")


if __name__ == "__main__":
    test_write_report_formats_metrics_and_citations()
    print("ok")
