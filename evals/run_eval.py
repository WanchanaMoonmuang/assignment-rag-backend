"""RAGAS evaluation harness for the RAG backend.

Usage:
    uv run --extra eval python -m evals.run_eval [--mode ingest|reuse] [options]

See evals/README.md for the golden-set format and the two run modes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.extraction import extract
from app.main import (
    build_prompt,
    chunk_col,
    doc_col,
    generate_answer,
    make_id,
    retrieve_chunks,
    validated_content_type,
)
from app.rag import GeminiClient
from app.settings import Settings, get_settings
from app.worker import build_chunk_documents
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    # Unlike embed_content (app/rag.py), generate_content has no retry/backoff in
    # production; 30 sequential eval questions can exhaust the per-minute generate
    # quota just like embed_content did before PR #21, so the harness retries locally.
    from google.genai import errors as genai_errors

    return isinstance(exc, genai_errors.APIError) and exc.code in _RETRYABLE_STATUS_CODES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES_DIR = REPO_ROOT / "samples"
DEFAULT_GOLDEN_SET = REPO_ROOT / "evals" / "golden_set.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "evals" / "results"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def load_golden_set(path: Path) -> dict[str, list[dict[str, str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    golden: dict[str, list[dict[str, str]]] = {}
    for filename, rows in raw.items():
        filled = [
            row
            for row in rows
            if str(row.get("question", "")).strip() and str(row.get("ground_truth", "")).strip()
        ]
        skipped = len(rows) - len(filled)
        if skipped:
            print(f"[golden_set] {filename}: skipping {skipped} unfilled stub row(s)")
        if filled:
            golden[filename] = filled
    return golden


async def ingest_samples(
    db: Any,
    settings: Settings,
    gemini: GeminiClient,
    samples_dir: Path,
    filenames: list[str],
    run_id: str,
) -> tuple[list[str], tuple[str, str]]:
    """Ingest each sample file via the real extract -> embed -> store path.

    Returns (document_ids ingested, (canary_question, canary_chunk_id)) where the
    canary is a short excerpt of the very first chunk, used to detect when the
    Atlas search/vector indexes have caught up with these inserts.
    """
    document_ids: list[str] = []
    canary: tuple[str, str] | None = None
    created_at = now_utc()

    for filename in filenames:
        data = (samples_dir / filename).read_bytes()
        suffix = Path(filename).suffix
        content_type = validated_content_type(suffix, data)
        chunks = extract(data, suffix, settings.rag_chunk_size, settings.rag_chunk_overlap)
        if not chunks:
            print(f"[ingest] {filename}: no extractable chunks, skipping")
            continue

        print(f"[ingest] {filename}: embedding {len(chunks)} chunk(s)...")
        embeddings = await gemini.embed_texts(
            [chunk.content for chunk in chunks], task_type="RETRIEVAL_DOCUMENT"
        )

        document_id = make_id("doc")
        document = {
            "_id": document_id,
            "document_name": filename,
            "source": "file_upload",
            "source_kind": "file",
            "source_format": suffix.removeprefix(".").lower(),
            "content_type": content_type,
            "byte_size": len(data),
            "chunks_count": len(chunks),
            "metadata": {"eval_run": run_id},
            "created_at": created_at,
            "updated_at": created_at,
        }
        job = {"document_id": document_id, "document_name": filename}
        chunk_documents = build_chunk_documents(job, chunks, embeddings, document, created_at)

        await doc_col(db, settings).insert_one(document)
        await chunk_col(db, settings).insert_many(chunk_documents)
        document_ids.append(document_id)

        if canary is None:
            words = chunk_documents[0]["content"].split()
            canary = (" ".join(words[:8]), chunk_documents[0]["_id"])

    if canary is None:
        raise RuntimeError("No sample file produced any chunks to ingest")
    return document_ids, canary


async def wait_for_index_consistency(
    db: Any,
    settings: Settings,
    gemini: GeminiClient,
    canary_question: str,
    canary_chunk_id: str,
    timeout_seconds: float = 90.0,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        chunks = await retrieve_chunks(canary_question, 5, db, settings, gemini)
        if any(chunk["chunk_id"] == canary_chunk_id for chunk in chunks):
            print(f"[index-wait] search index caught up after {attempt} attempt(s)")
            return
        if asyncio.get_event_loop().time() >= deadline:
            raise RuntimeError(
                "Atlas search/vector index did not catch up with freshly ingested "
                f"chunks within {timeout_seconds:.0f}s"
            )
        await asyncio.sleep(3)


async def cleanup_eval_docs(db: Any, settings: Settings, document_ids: list[str]) -> None:
    if not document_ids:
        return
    await chunk_col(db, settings).delete_many({"document_id": {"$in": document_ids}})
    await doc_col(db, settings).delete_many({"_id": {"$in": document_ids}})
    print(f"[cleanup] removed {len(document_ids)} eval document(s) and their chunks")


async def answer_question(
    db: Any, settings: Settings, gemini: GeminiClient, question: str
) -> tuple[list[str], str]:
    chunks = await retrieve_chunks(question, settings.rag_top_k, db, settings, gemini)
    prompt, _ = build_prompt(chunks, question, [], settings.generation_context_token_budget)
    answer = ""
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_gemini_error),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    ):
        with attempt:
            answer, _tool_activity, _usage = await generate_answer(gemini, prompt)
    return [chunk["content"] for chunk in chunks], answer or ""


# langchain_google_vertexai builds the API hostname as f"{location}-aiplatform.googleapis.com",
# which is only valid for concrete regions (e.g. "us-central1"); Vertex multi-region
# aliases ("us", "eu") instead live at the bare global host, so those need an explicit
# override or every judge/embedding call gets an "Invalid hostname" 400.
_MULTI_REGION_ALIASES = {"us", "eu"}


def build_judge(settings: Settings) -> tuple[Any, Any]:
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    project = settings.google_cloud_project or settings.gcp_project_id
    location = settings.google_cloud_location
    api_endpoint = "aiplatform.googleapis.com" if location in _MULTI_REGION_ALIASES else None
    llm = LangchainLLMWrapper(
        ChatVertexAI(
            model=settings.eval_judge_model,
            project=project,
            location=location,
            api_endpoint=api_endpoint,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        VertexAIEmbeddings(
            model=settings.gemini_embedding_model,
            project=project,
            location=location,
            dimensions=settings.gemini_embedding_dimensions,
        )
    )
    return llm, embeddings


def run_ragas(
    rows: list[dict[str, Any]],
    settings: Settings,
    include_answer_relevancy: bool,
) -> Any:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    judge_llm, judge_embeddings = build_judge(settings)
    metrics: list[Any] = [
        Faithfulness(llm=judge_llm),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]
    if include_answer_relevancy:
        metrics.append(ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings))

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=row["question"],
                retrieved_contexts=row["contexts"],
                response=row["answer"],
                reference=row["ground_truth"],
            )
            for row in rows
        ]
    )
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=settings.eval_ragas_max_workers, max_wait=90),
    )


def write_report(
    rows: list[dict[str, Any]], result: Any, output_dir: Path, settings: Settings
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    frame = result.to_pandas()

    models = {
        "serving_model": settings.gemini_model,
        "judge_model": settings.eval_judge_model,
        "embedding_model": settings.gemini_embedding_model,
        "embedding_dimensions": settings.gemini_embedding_dimensions,
    }
    retrieval = {
        "rag_top_k": settings.rag_top_k,
        "rag_chunk_size": settings.rag_chunk_size,
        "rag_chunk_overlap": settings.rag_chunk_overlap,
        "mongodb_vector_index": settings.mongodb_vector_index,
        "mongodb_search_index": settings.mongodb_search_index,
        # $rankFusion weights are hardcoded equal in app/main.py:retrieve_chunks, not a setting.
        "rankfusion_weights": {"vector": 1, "lexical": 1},
        "generation_context_token_budget": settings.generation_context_token_budget,
    }

    # ragas' output columns are the dataset input columns plus one column per metric;
    # metric columns are whatever's left after the known input columns.
    known_input_cols = {"user_input", "retrieved_contexts", "response", "reference"}
    metric_names = [col for col in frame.columns if col not in known_input_cols]
    # A metric can legitimately return NaN for a single row (e.g. faithfulness when no
    # verifiable statements were extracted from the answer); pandas' mean() already
    # skips NaN, unlike statistics.fmean(), which would silently poison the aggregate.
    aggregates = {name: frame[name].mean() for name in metric_names}

    json_path = output_dir / f"{timestamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "models": models,
                "retrieval": retrieval,
                "aggregates": aggregates,
                "rows": [
                    {
                        "filename": rows[i]["filename"],
                        "question": rows[i]["question"],
                        "answer": rows[i]["answer"],
                        **{name: frame[name].iloc[i] for name in metric_names},
                    }
                    for i in range(len(rows))
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    md_lines = ["# RAGAS eval results", "", "## Models", ""]
    md_lines.append(f"- **Serving model**: {models['serving_model']}")
    md_lines.append(f"- **Judge model**: {models['judge_model']}")
    md_lines.append(
        f"- **Embedding model**: {models['embedding_model']} ({models['embedding_dimensions']} dims)"
    )
    md_lines += ["", "## Retrieval config", ""]
    md_lines.append(f"- **rag_top_k**: {retrieval['rag_top_k']}")
    md_lines.append(
        f"- **Chunking**: size={retrieval['rag_chunk_size']}, overlap={retrieval['rag_chunk_overlap']}"
    )
    md_lines.append(f"- **Vector index**: {retrieval['mongodb_vector_index']}")
    md_lines.append(f"- **Search index**: {retrieval['mongodb_search_index']}")
    md_lines.append(f"- **$rankFusion weights**: {retrieval['rankfusion_weights']}")
    md_lines.append(
        f"- **Generation context token budget**: {retrieval['generation_context_token_budget']}"
    )
    md_lines += ["", "## Aggregate scores", ""]
    for name, value in aggregates.items():
        md_lines.append(f"- **{name}**: {value:.4f}")
    md_lines += ["", "## Per-question scores", "", f"| file | question | {' | '.join(metric_names)} |"]
    md_lines.append(f"|---|---|{'---|' * len(metric_names)}")
    for i, row in enumerate(rows):
        scores = " | ".join(f"{frame[name].iloc[i]:.3f}" for name in metric_names)
        question = row["question"].replace("|", "\\|")
        md_lines.append(f"| {row['filename']} | {question} | {scores} |")

    md_lines += ["", "## Per-question detail", ""]
    for i, row in enumerate(rows):
        scores = ", ".join(f"{name}={frame[name].iloc[i]:.3f}" for name in metric_names)
        md_lines.append(f"### {row['filename']} — {scores}")
        md_lines.append(f"**Question:** {row['question']}")
        md_lines.append(f"**Model's answer:** {row['answer']}")
        md_lines.append(f"**Ground truth:** {row['ground_truth']}")
        md_lines.append("")

    md_path = output_dir / f"{timestamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return json_path


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    from motor.motor_asyncio import AsyncIOMotorClient

    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured")
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]
    gemini = GeminiClient(settings)

    golden_set = load_golden_set(args.golden_set)
    if not golden_set:
        raise RuntimeError(f"No filled-in rows found in {args.golden_set}")

    document_ids: list[str] = []
    try:
        if args.mode == "ingest":
            existing = await chunk_col(db, settings).count_documents({})
            if existing and not args.force:
                raise RuntimeError(
                    f"{settings.mongodb_chunk_collection} already has {existing} chunk(s); "
                    "pass --force to ingest anyway, or use --mode reuse."
                )
            document_ids, (canary_question, canary_chunk_id) = await ingest_samples(
                db, settings, gemini, args.samples_dir, list(golden_set.keys()), run_id=make_id("eval")
            )
            await wait_for_index_consistency(db, settings, gemini, canary_question, canary_chunk_id)
        else:
            print("[reuse] skipping ingestion; scoring against the corpus already in the database")

        rows: list[dict[str, Any]] = []
        for filename, golden_rows in golden_set.items():
            for golden_row in golden_rows:
                print(f"[answer] {filename}: {golden_row['question'][:80]}")
                contexts, answer = await answer_question(db, settings, gemini, golden_row["question"])
                rows.append(
                    {
                        "filename": filename,
                        "question": golden_row["question"],
                        "ground_truth": golden_row["ground_truth"],
                        "contexts": contexts,
                        "answer": answer,
                    }
                )

        print(f"[ragas] scoring {len(rows)} question(s) with judge={settings.eval_judge_model}...")
        result = run_ragas(rows, settings, include_answer_relevancy=not args.no_answer_relevancy)
        print(result)
        report_path = write_report(rows, result, args.output_dir, settings)
        print(f"[report] wrote {report_path}")
    finally:
        if args.mode == "ingest" and not args.keep:
            await cleanup_eval_docs(db, settings, document_ids)
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ingest", "reuse"], default="ingest")
    parser.add_argument("--force", action="store_true", help="Ingest even if chunks already exist")
    parser.add_argument("--keep", action="store_true", help="Don't delete eval documents afterward")
    parser.add_argument("--no-answer-relevancy", action="store_true")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
