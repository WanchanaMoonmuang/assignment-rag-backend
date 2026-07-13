from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pymongo import ReturnDocument

from app.extraction import ExtractedChunk, ExtractionError, extract, extract_text
from app.rag import GeminiClient, make_id
from app.settings import Settings, get_settings
from app.storage import delete_object, download_object

logger = logging.getLogger(__name__)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def job_filter(job: dict[str, Any]) -> dict[str, Any]:
    return {"_id": job["_id"], "status": "processing", "lease_token": job["lease_token"]}


def cleanup_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "_id": make_id("job"),
        "document_id": job["document_id"],
        "document_name": job["document_name"],
        "source_kind": "cleanup",
        "metadata": {},
        "object_name": job["object_name"],
        "status": "cleanup_pending",
        "stage": "cleanup",
        "cleanup_attempts": 0,
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }


async def mark_job_failed(
    db: Any,
    settings: Settings,
    job: dict[str, Any],
    error: dict[str, str],
    terminal: bool,
    failure_filter: dict[str, Any],
) -> bool:
    jobs = db[settings.mongodb_ingestion_job_collection]
    update = {
        "$set": {
            "status": "failed" if terminal else "queued",
            "stage": "failed",
            "error": error,
            "updated_at": now_utc(),
        },
        "$unset": {"lease_owner": "", "lease_token": "", "lease_expires_at": ""},
    }
    if not terminal or job.get("source_kind") != "file" or not job.get("object_name"):
        return (await jobs.update_one(failure_filter, update)).modified_count == 1

    session = await db.client.start_session()
    async with session:
        async with session.start_transaction():
            result = await jobs.update_one(failure_filter, update, session=session)
            if result.modified_count != 1:
                return False
            await jobs.insert_one(cleanup_job(job), session=session)
    return True


async def terminalize_expired_jobs(db: Any, settings: Settings, now: datetime) -> None:
    jobs = db[settings.mongodb_ingestion_job_collection]
    cursor = jobs.find(
        {
            "status": "processing",
            "lease_expires_at": {"$lt": now},
            "attempts": {"$gte": settings.ingestion_job_max_attempts},
        }
    )
    for job in await cursor.to_list(length=50):
        if (
            job.get("status") != "processing"
            or job.get("lease_expires_at") is None
            or job["lease_expires_at"] >= now
            or int(job.get("attempts") or 0) < settings.ingestion_job_max_attempts
        ):
            continue
        await mark_job_failed(
            db,
            settings,
            job,
            {"code": "attempts_exhausted", "message": "Unable to process ingestion"},
            True,
            {
                "_id": job["_id"],
                "status": "processing",
                "lease_expires_at": {"$lt": now},
            },
        )


async def claim_job(db: Any, settings: Settings, worker_id: str) -> dict[str, Any] | None:
    jobs = db[settings.mongodb_ingestion_job_collection]
    now = now_utc()
    await terminalize_expired_jobs(db, settings, now)
    return await jobs.find_one_and_update(
        {
            "$or": [
                {"status": "queued"},
                {
                    "status": "processing",
                    "lease_expires_at": {"$lt": now},
                    "attempts": {"$lt": settings.ingestion_job_max_attempts},
                },
            ]
        },
        {
            "$set": {
                "status": "processing",
                "stage": "converting",
                "lease_owner": worker_id,
                "lease_token": make_id("lease"),
                "lease_expires_at": now + timedelta(seconds=settings.ingestion_job_lease_seconds),
                "updated_at": now,
            },
            "$inc": {"attempts": 1},
        },
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )

async def renew_lease(db: Any, settings: Settings, job: dict[str, Any]) -> None:
    jobs = db[settings.mongodb_ingestion_job_collection]
    while True:
        await asyncio.sleep(max(1, settings.ingestion_job_lease_seconds // 2))
        result = await jobs.update_one(
            job_filter(job),
            {
                "$set": {
                    "lease_expires_at": now_utc()
                    + timedelta(seconds=settings.ingestion_job_lease_seconds),
                    "updated_at": now_utc(),
                }
            },
        )
        if result.modified_count != 1:
            return


async def set_stage(db: Any, settings: Settings, job: dict[str, Any], stage: str) -> None:
    await db[settings.mongodb_ingestion_job_collection].update_one(
        job_filter(job), {"$set": {"stage": stage, "updated_at": now_utc()}}
    )


async def publish_job(
    db: Any,
    settings: Settings,
    gemini: GeminiClient,
    job: dict[str, Any],
    chunks: list[ExtractedChunk],
    document: dict[str, Any],
) -> None:
    if not chunks:
        raise ExtractionError("Document does not contain extractable text")
    await set_stage(db, settings, job, "embedding")
    embeddings = await gemini.embed_texts([chunk.content for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count did not match chunk count")

    created_at = now_utc()
    document_id = job["document_id"]
    metadata = job.get("metadata") or {}
    document.update(
        {
            "_id": document_id,
            "chunks_count": len(chunks),
            "metadata": metadata,
            "created_at": created_at,
            "updated_at": created_at,
        }
    )
    chunk_documents = [
        {
            "_id": make_id("chunk"),
            "document_id": document_id,
            "document_name": job["document_name"],
            "chunk_index": index,
            "content": chunk.content,
            "embedding": embedding,
            "metadata": metadata,
            "source_format": document["source_format"],
            "chunk_type": chunk.chunk_type,
            "location": chunk.location,
            "created_at": created_at,
        }
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1)
    ]

    await set_stage(db, settings, job, "finalizing")
    jobs = db[settings.mongodb_ingestion_job_collection]
    session = await db.client.start_session()
    async with session:
        async with session.start_transaction():
            await db[settings.mongodb_document_collection].insert_one(document, session=session)
            await db[settings.mongodb_chunk_collection].insert_many(chunk_documents, session=session)
            result = await jobs.update_one(
                job_filter(job),
                {
                    "$set": {
                        "status": "completed",
                        "stage": "finalizing",
                        "completed_at": now_utc(),
                        "updated_at": now_utc(),
                    },
                    "$unset": {"lease_owner": "", "lease_token": "", "lease_expires_at": ""},
                },
                session=session,
            )
            if result.modified_count != 1:
                raise RuntimeError("Ingestion lease was lost")


async def process_text_job(db: Any, settings: Settings, gemini: GeminiClient, job: dict[str, Any]) -> None:
    content = str(job["content"])
    await set_stage(db, settings, job, "chunking")
    chunks = extract_text(
        content.encode("utf-8"), settings.rag_chunk_size, settings.rag_chunk_overlap
    )
    await publish_job(
        db,
        settings,
        gemini,
        job,
        chunks,
        {
            "document_name": job["document_name"],
            "source": "plain_text",
            "source_kind": "text",
            "source_format": "txt",
            "content_type": "text/plain; charset=utf-8",
            "byte_size": len(content.encode("utf-8")),
        },
    )


async def process_file_job(db: Any, settings: Settings, gemini: GeminiClient, job: dict[str, Any]) -> None:
    await set_stage(db, settings, job, "converting")
    try:
        data = await download_object(settings, job["object_name"])
    except Exception as exc:
        raise ExtractionError("Original file could not be downloaded") from exc
    await set_stage(db, settings, job, "extracting")
    suffix = Path(job["document_name"]).suffix
    chunks = extract(data, suffix, settings.rag_chunk_size, settings.rag_chunk_overlap)
    await publish_job(
        db,
        settings,
        gemini,
        job,
        chunks,
        {
            "document_name": job["document_name"],
            "source": "file_upload",
            "source_kind": "file",
            "source_format": suffix.removeprefix(".").lower(),
            "content_type": job["content_type"],
            "byte_size": len(data),
            "original_object_name": job["object_name"],
        },
    )


async def process_job(db: Any, settings: Settings, gemini: GeminiClient, job: dict[str, Any]) -> None:
    heartbeat = asyncio.create_task(renew_lease(db, settings, job))
    try:
        try:
            process = process_text_job if job["source_kind"] == "text" else process_file_job
            await asyncio.wait_for(
                process(db, settings, gemini, job),
                timeout=settings.ingestion_processing_timeout_seconds,
            )
        except TimeoutError:
            error = {"code": "processing_timeout", "message": "Ingestion processing timed out"}
        except ExtractionError as exc:
            error = {"code": "extraction_failed", "message": str(exc)}
        except Exception:
            error = {"code": "ingestion_failed", "message": "Unable to process ingestion"}
        else:
            return

        await mark_job_failed(
            db,
            settings,
            job,
            error,
            int(job.get("attempts") or 0) >= settings.ingestion_job_max_attempts,
            job_filter(job),
        )
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat

async def sweep_cleanup(db: Any, settings: Settings) -> None:
    jobs = db[settings.mongodb_ingestion_job_collection]
    cursor = jobs.find({"status": "cleanup_pending"})
    for job in await cursor.to_list(length=50):
        attempts = int(job.get("cleanup_attempts") or 0) + 1
        try:
            await delete_object(settings, job["object_name"])
            await jobs.update_one(
                {"_id": job["_id"], "status": "cleanup_pending"},
                {"$set": {"status": "cleanup_completed", "updated_at": now_utc()}},
            )
        except Exception:
            terminal = attempts >= 10
            if terminal:
                logger.error("gcs_cleanup_failed job_id=%s attempts=%s", job["_id"], attempts)
            await jobs.update_one(
                {"_id": job["_id"], "status": "cleanup_pending"},
                {
                    "$set": {
                        "status": "cleanup_failed" if terminal else "cleanup_pending",
                        "cleanup_attempts": attempts,
                        "updated_at": now_utc(),
                    }
                },
            )


async def run() -> None:
    from motor.motor_asyncio import AsyncIOMotorClient

    settings = get_settings()
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured")
    client = AsyncIOMotorClient(settings.mongodb_uri)
    try:
        db = client[settings.mongodb_database]
        gemini = GeminiClient(settings)
        next_cleanup = 0.0
        while True:
            if asyncio.get_running_loop().time() >= next_cleanup:
                await sweep_cleanup(db, settings)
                next_cleanup = asyncio.get_running_loop().time() + 300
            job = await claim_job(db, settings, "worker")
            if job is None:
                await asyncio.sleep(1)
                continue
            await process_job(db, settings, gemini, job)
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(run())
