from __future__ import annotations

from contextlib import asynccontextmanager
import io
import json
from urllib.parse import quote
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, AsyncIterator
import zipfile

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pymongo.errors import OperationFailure

from app.observability import emit
from app.auth import create_access_token, credentials_match, get_current_username
from app.rag import (
    FALLBACK_ANSWER,
    GeminiClient,
    build_prompt,
    clamp_top_k,
    make_id,
    public_sources,
    estimated_tokens,
    question_fits_budget,
    sse,
)
from app.schemas import ChatRequest, IngestRequest, IngestionTextRequest, LoginRequest
from app.settings import Settings, get_settings
from app.storage import delete_object, download_object, upload_object


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def configure_gemini_client(app: FastAPI, settings: Settings) -> None:
    try:
        app.state.gemini = GeminiClient(settings)
        app.state.gemini_error = None
    except RuntimeError as exc:
        app.state.gemini = None
        app.state.gemini_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    if settings.mongodb_uri:
        from motor.motor_asyncio import AsyncIOMotorClient

        app.state.mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
        app.state.db = app.state.mongo_client[settings.mongodb_database]
    configure_gemini_client(app, settings)
    try:
        yield
    finally:
        if hasattr(app.state, "mongo_client"):
            app.state.mongo_client.close()


app = FastAPI(title="RAG Chatbot Backend", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


router = APIRouter(prefix="/api")
SUPPORTED_FILE_EXTENSIONS = ("txt", "pdf", "docx", "csv", "json")


def get_db(request: Request) -> Any:
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="MONGODB_URI is not configured")
    return db


def get_gemini(request: Request) -> GeminiClient:
    gemini = getattr(request.app.state, "gemini", None)
    if gemini is None:
        detail = getattr(request.app.state, "gemini_error", None) or "Gemini client is not configured"
        raise HTTPException(status_code=500, detail=detail)
    return gemini


def doc_col(db: Any, settings: Settings) -> Any:
    return db[settings.mongodb_document_collection]


def chunk_col(db: Any, settings: Settings) -> Any:
    return db[settings.mongodb_chunk_collection]


def conv_col(db: Any, settings: Settings) -> Any:
    return db[settings.mongodb_conversation_collection]


def job_col(db: Any, settings: Settings) -> Any:
    return db[settings.mongodb_ingestion_job_collection]


def serialize_document(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    return {
        "document_id": document["_id"],
        "document_name": document["document_name"],
        "source": document.get("source") or metadata.get("source") or "plain_text",
        "chunks_count": document.get("chunks_count", 0),
        "created_at": document.get("created_at"),
        "updated_at": document.get("updated_at"),
    }


def serialize_conversation(conversation: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_id": conversation["_id"],
        "title": conversation["title"],
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
        "last_message_preview": conversation.get("last_message_preview"),
    }


def title_from_question(question: str) -> str:
    return question[:80]


async def ensure_conversation(
    db: Any, settings: Settings, conversation_id: str | None, question: str
) -> dict[str, Any]:
    conversations = conv_col(db, settings)
    if conversation_id:
        conversation = await conversations.find_one({"_id": conversation_id})
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    created_at = now_utc()
    conversation = {
        "_id": make_id("conv"),
        "title": title_from_question(question),
        "created_at": created_at,
        "updated_at": created_at,
        "last_message_preview": question[:120],
        "messages": [],
    }
    await conversations.insert_one(conversation)
    return conversation


async def append_message(
    db: Any,
    settings: Settings,
    conversation_id: str,
    message: dict[str, Any],
    preview: str,
) -> None:
    await conv_col(db, settings).update_one(
        {"_id": conversation_id},
        {
            "$push": {"messages": message},
            "$set": {"updated_at": now_utc(), "last_message_preview": preview[:160]},
        },
    )


def history_context_messages(conversation: dict[str, Any], window: int) -> list[dict[str, Any]]:
    if window <= 0:
        return []
    messages = conversation.get("messages", [])
    return messages[-window:]


async def prepare_chat(
    payload: ChatRequest,
    db: Any,
    settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not question_fits_budget(payload.question, settings.generation_context_token_budget):
        raise HTTPException(status_code=422, detail="Question exceeds the configured context budget")
    conversation = await ensure_conversation(db, settings, payload.conversation_id, payload.question)
    history_messages = history_context_messages(conversation, settings.history_context_window)
    await append_message(
        db,
        settings,
        conversation["_id"],
        {"role": "user", "content": payload.question, "created_at": now_utc()},
        payload.question,
    )
    return conversation, history_messages


async def retrieve_chunks(
    question: str,
    top_k: int,
    db: Any,
    settings: Settings,
    gemini: GeminiClient,
) -> list[dict[str, Any]]:
    if top_k == 0:
        return []
    if not settings.mongodb_vector_index:
        raise HTTPException(status_code=500, detail="MONGODB_VECTOR_INDEX is not configured")
    [query_embedding] = await gemini.embed_texts([question], task_type="RETRIEVAL_QUERY")
    candidate_limit = max(top_k * 4, 20)
    pipeline = [
        {
            "$rankFusion": {
                "input": {"pipelines": {
                    "vector": [{"$vectorSearch": {
                        "index": settings.mongodb_vector_index,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": max(candidate_limit * 20, 100),
                        "limit": candidate_limit,
                    }}],
                    "lexical": [
                        {"$search": {"index": settings.mongodb_search_index,
                          "text": {"query": question, "path": "content"}}},
                        {"$limit": candidate_limit},
                    ],
                }},
                "combination": {"weights": {"vector": 1, "lexical": 1}},
                "scoreDetails": True,
            }
        },
        {"$limit": top_k},
        {
            "$project": {
                "_id": 1,
                "document_id": 1,
                "document_name": 1,
                "content": 1,
                "source_format": 1,
                "chunk_type": 1,
                "location": 1,
                "metadata": 1,
                "score": {"$meta": "score"},
            }
        },
    ]
    try:
        rows = await chunk_col(db, settings).aggregate(pipeline).to_list(length=top_k)
    except OperationFailure as exc:
        raise HTTPException(status_code=503, detail="MongoDB search indexes are unavailable") from exc
    chunks = []
    for row in rows:
        score = float(row.get("score") or 0)
        content = row.get("content", "")
        chunks.append(
            {
                "document_id": row["document_id"],
                "document_name": row["document_name"],
                "chunk_id": row["_id"],
                "content": content,
                "snippet": content[:500],
                "score": score,
                "source_format": row.get("source_format"),
                "chunk_type": row.get("chunk_type"),
                "location": row.get("location"),
                "metadata": row.get("metadata"),
            }
        )
    return chunks


async def generate_answer(
    gemini: GeminiClient,
    prompt: str,
) -> tuple[str, list[dict[str, Any]], dict[str, int] | None]:
    generate_with_calculator = getattr(gemini, "generate_with_calculator", None)
    if generate_with_calculator is None:
        return await gemini.generate(prompt), [], None
    result = await generate_with_calculator(prompt)
    if len(result) == 3:
        return result
    answer, activity = result
    return answer, activity, None



async def completed_answer(
    payload: ChatRequest,
    db: Any,
    settings: Settings,
    gemini: GeminiClient,
    history_messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    started = perf_counter()
    chunks = await retrieve_chunks(
        payload.question, clamp_top_k(payload.top_k, settings.rag_top_k), db, settings, gemini
    )
    retrieval_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    prompt, prompt_chunks = build_prompt(
        chunks, payload.question, history_messages, settings.generation_context_token_budget
    )
    prompt_build_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    answer, tool_activity, usage = await generate_answer(gemini, prompt)
    model_ms = (perf_counter() - started) * 1000
    answer = answer or FALLBACK_ANSWER
    tokens = usage or {
        "input_tokens": estimated_tokens(prompt),
        "output_tokens": estimated_tokens(answer),
        "total_tokens": estimated_tokens(prompt) + estimated_tokens(answer),
    }
    return answer, public_sources(prompt_chunks), tool_activity, {
        "retrieval_ms": round(retrieval_ms, 2),
        "prompt_build_ms": round(prompt_build_ms, 2),
        "model_first_token_ms": None,
        "model_total_ms": round(model_ms, 2),
        **tokens,
        "token_source": "provider" if usage else "estimated",
        "retrieved_chunks": len(chunks),
        "scores": [chunk["score"] for chunk in chunks],
    }


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/config")
async def config(
    username: str = Depends(get_current_username),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return {
        "rag_top_k": {"default": settings.rag_top_k, "min": 0, "max": 20},
        "max_upload_bytes": settings.max_upload_bytes,
        "supported_file_extensions": list(SUPPORTED_FILE_EXTENSIONS),
    }


@router.post("/auth/login")
async def login(
    payload: LoginRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    if not credentials_match(payload.username, payload.password, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return {
        "access_token": create_access_token(payload.username, settings),
        "token_type": "bearer",
    }


@router.get("/auth/me")
async def me(username: str = Depends(get_current_username)) -> dict[str, str]:
    return {"username": username}


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    payload: IngestRequest,
    username: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return await create_text_ingestion(
        IngestionTextRequest(
            document_name=payload.document_name,
            content=payload.content,
            metadata=payload.metadata,
        ),
        username,
        db,
        settings,
    )


def serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["_id"],
        "document_id": job["document_id"],
        "document_name": job["document_name"],
        "status": job["status"],
        "stage": job.get("stage"),
        "error": job.get("error"),
    }


def validated_content_type(suffix: str, data: bytes) -> str:
    try:
        if suffix == ".pdf":
            if not data.startswith(b"%PDF-"):
                raise ValueError
            return "application/pdf"
        if suffix == ".docx":
            if not data.startswith(b"PK"):
                raise ValueError
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise ValueError
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if suffix == ".json":
            json.loads(data.decode("utf-8-sig"))
            return "application/json"
        if suffix in {".txt", ".csv"}:
            data.decode("utf-8-sig")
            return "text/plain" if suffix == ".txt" else "text/csv"
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, zipfile.BadZipFile):
        pass
    raise HTTPException(status_code=415, detail="File content does not match its extension")


def new_job(
    document_id: str,
    document_name: str,
    source_kind: str,
    metadata: dict[str, Any],
    **values: Any,
) -> dict[str, Any]:
    created_at = now_utc()
    return {
        "_id": make_id("job"),
        "document_id": document_id,
        "document_name": document_name,
        "source_kind": source_kind,
        "metadata": metadata,
        "status": "queued",
        "stage": "queued",
        "attempts": 0,
        "created_at": created_at,
        "updated_at": created_at,
        **values,
    }


@router.post("/ingestions/text", status_code=status.HTTP_202_ACCEPTED)
async def create_text_ingestion(
    payload: IngestionTextRequest,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if len(payload.content.encode("utf-8")) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Document content exceeds 20 MiB limit")
    if not payload.content.strip():
        raise HTTPException(status_code=422, detail="Document content is required")

    job = new_job(
        make_id("doc"),
        payload.document_name,
        "text",
        payload.metadata or {},
        content=payload.content,
    )
    await job_col(db, settings).insert_one(job)
    return serialize_job(job)


@router.post("/ingestions/file", status_code=status.HTTP_202_ACCEPTED)
async def create_file_ingestion(
    file: UploadFile = File(...),
    metadata_json: str | None = Form(default=None),
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=422, detail="File name is required")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx", ".csv", ".json"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Metadata must be valid JSON") from exc
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=422, detail="Metadata must be an object")

    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds 20 MiB limit")
    content_type = validated_content_type(suffix, data)

    document_id = make_id("doc")
    object_name = f"originals/{document_id}/{filename}"
    job = new_job(
        document_id,
        filename,
        "file",
        metadata,
        object_name=object_name,
        content_type=content_type,
    )
    # Insert the job only once the upload outcome is known, so it never becomes
    # claimable before the object actually exists in GCS (a claim-before-upload
    # race otherwise causes a spurious first-attempt "could not be downloaded").
    try:
        await upload_object(settings, object_name, data, content_type)
    except Exception:
        job["status"] = "failed"
        job["stage"] = "failed"
        job["error"] = {"code": "original_upload_failed", "message": "Original file upload failed"}
        await job_col(db, settings).insert_one(job)
        await schedule_cleanup(db, settings, document_id, filename, object_name)
        return serialize_job(job)
    await job_col(db, settings).insert_one(job)
    return serialize_job(job)


@router.get("/ingestions/{job_id}")
async def get_ingestion_job(
    job_id: str,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    job = await job_col(db, settings).find_one({"_id": job_id})
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return serialize_job(job)



@router.get("/documents")
async def list_documents(
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    cursor = doc_col(db, settings).find({}).sort("created_at", -1)
    documents = await cursor.to_list(length=None)
    return {"documents": [serialize_document(document) for document in documents]}


@router.get("/documents/{document_id}/chunks/{chunk_id}")
async def get_cited_chunk(
    document_id: str,
    chunk_id: str,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    chunk = await chunk_col(db, settings).find_one({"_id": chunk_id, "document_id": document_id})
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    index = int(chunk.get("chunk_index") or 0)
    rows = await chunk_col(db, settings).find(
        {"document_id": document_id, "chunk_index": {"$gte": index - 2, "$lte": index + 2}}
    ).sort("chunk_index", 1).to_list(length=5)
    return {
        "document_id": document_id,
        "chunk": {
            "chunk_id": chunk["_id"],
            "content": chunk.get("content", ""),
            "chunk_index": index,
            "location": chunk.get("location"),
            "metadata": chunk.get("metadata") or {},
        },
        "neighbors": [
            {
                "chunk_id": row["_id"],
                "content": row.get("content", ""),
                "chunk_index": row.get("chunk_index"),
                "location": row.get("location"),
            }
            for row in rows
        ],
    }


@router.get("/documents/{document_id}/file")
async def get_original_file(
    document_id: str,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    document = await doc_col(db, settings).find_one({"_id": document_id})
    if document is None or not document.get("original_object_name"):
        raise HTTPException(status_code=404, detail="Original file not found")
    try:
        content = await download_object(settings, document["original_object_name"])
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Original file not found") from exc
    filename = quote(document["document_name"], safe="")
    return Response(
        content=content,
        media_type=document.get("content_type") or "application/octet-stream",
        headers={"Content-Disposition": "attachment; filename*=UTF-8''" + filename},
    )






async def schedule_cleanup(
    db: Any, settings: Settings, document_id: str, document_name: str, object_name: str
) -> None:
    job = new_job(document_id, document_name, "cleanup", {}, object_name=object_name)
    job["status"] = "cleanup_pending"
    job["stage"] = "cleanup"
    job["cleanup_attempts"] = 0
    await job_col(db, settings).insert_one(job)


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    documents = doc_col(db, settings)
    chunks = chunk_col(db, settings)
    cleanup_job: dict[str, Any] | None = None
    session = await db.client.start_session()
    try:
        async with session:
            async with session.start_transaction():
                document = await documents.find_one({"_id": document_id}, session=session)
                if document is None:
                    raise HTTPException(status_code=404, detail="Document not found")
                object_name = document.get("original_object_name")
                if object_name:
                    cleanup_job = new_job(
                        document_id, document["document_name"], "cleanup", {}, object_name=object_name
                    )
                    cleanup_job["status"] = "cleanup_pending"
                    cleanup_job["stage"] = "cleanup"
                    cleanup_job["cleanup_attempts"] = 0
                    await job_col(db, settings).insert_one(cleanup_job, session=session)
                deleted = await chunks.delete_many({"document_id": document_id}, session=session)
                metadata_deleted = await documents.delete_one({"_id": document_id}, session=session)
                if metadata_deleted.deleted_count != 1:
                    raise RuntimeError("document metadata delete failed")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to delete document") from exc

    if cleanup_job:
        try:
            await delete_object(settings, cleanup_job["object_name"])
            await job_col(db, settings).update_one(
                {"_id": cleanup_job["_id"], "status": "cleanup_pending"},
                {"$set": {"status": "cleanup_completed", "updated_at": now_utc()}},
            )
        except Exception:
            pass
    return {
        "status": "success",
        "document_id": document_id,
        "document_name": document["document_name"],
        "deleted_chunks": deleted.deleted_count,
    }


@router.get("/conversations")
async def list_conversations(
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    cursor = conv_col(db, settings).find({}, {"messages": 0}).sort("updated_at", -1)
    conversations = await cursor.to_list(length=None)
    return {"conversations": [serialize_conversation(item) for item in conversations]}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    conversation = await conv_col(db, settings).find_one({"_id": conversation_id})
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": conversation["_id"],
        "title": conversation["title"],
        "messages": conversation.get("messages", []),
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    result = await conv_col(db, settings).delete_one({"_id": conversation_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success", "conversation_id": conversation_id}


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    request_started = perf_counter()
    request_id = make_id("req")
    effective_top_k = clamp_top_k(payload.top_k, settings.rag_top_k)
    conversation_id: str | None = None
    try:
        conversation, history_messages = await prepare_chat(payload, db, settings)
        conversation_id = conversation["_id"]
        answer, sources, tool_activity, metrics = await completed_answer(
            payload, db, settings, gemini, history_messages
        )
        await append_message(
            db,
            settings,
            conversation_id,
            {"role": "assistant", "content": answer, "sources": sources, "tool_activity": tool_activity, "created_at": now_utc()},
            answer,
        )
        emit(
            "chat_metrics",
            request_id=request_id,
            conversation_id=conversation_id,
            provider=settings.gemini_provider,
            model=settings.gemini_model,
            requested_top_k=payload.top_k,
            effective_top_k=effective_top_k,
            tool_names=[item.get("name") for item in tool_activity],
            request_total_ms=round((perf_counter() - request_started) * 1000, 2),
            status="completed",
            **metrics,
        )
        return {"request_id": request_id, "conversation_id": conversation_id, "answer": answer, "sources": sources, "tool_activity": tool_activity}
    except Exception:
        emit(
            "chat_metrics",
            request_id=request_id,
            conversation_id=conversation_id,
            provider=settings.gemini_provider,
            model=settings.gemini_model,
            requested_top_k=payload.top_k,
            effective_top_k=effective_top_k,
            request_total_ms=round((perf_counter() - request_started) * 1000, 2),
            status="failed",
            error_code="chat_failed",
        )
        raise


@router.post("/chat/stream")
async def stream_chat(
    payload: ChatRequest,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    request_started = perf_counter()
    request_id = make_id("req")
    effective_top_k = clamp_top_k(payload.top_k, settings.rag_top_k)
    try:
        conversation, history_messages = await prepare_chat(payload, db, settings)
    except Exception:
        emit(
            "chat_metrics",
            request_id=request_id,
            conversation_id=None,
            provider=settings.gemini_provider,
            model=settings.gemini_model,
            requested_top_k=payload.top_k,
            effective_top_k=effective_top_k,
            request_total_ms=round((perf_counter() - request_started) * 1000, 2),
            status="failed",
            error_code="chat_preparation_failed",
        )
        raise
    async def events() -> AsyncIterator[str]:
        yield sse("conversation", {"conversation_id": conversation["_id"], "title": conversation["title"]})
        yield sse("metadata", {"request_id": request_id, "top_k": effective_top_k})
        try:
            retrieval_started = perf_counter()
            chunks = await retrieve_chunks(payload.question, effective_top_k, db, settings, gemini)
            retrieval_ms = (perf_counter() - retrieval_started) * 1000
            prompt_started = perf_counter()
            prompt, prompt_chunks = build_prompt(
                chunks, payload.question, history_messages, settings.generation_context_token_budget
            )
            prompt_build_ms = (perf_counter() - prompt_started) * 1000
            model_started = perf_counter()
            first_token_ms: float | None = None
            tool_activity: list[dict[str, Any]] = []
            answer_parts: list[str] = []
            stream_with_calculator = getattr(gemini, "stream_with_calculator", None)
            if stream_with_calculator is None:
                async for token in gemini.stream(prompt):
                    if first_token_ms is None:
                        first_token_ms = (perf_counter() - model_started) * 1000
                    answer_parts.append(token)
                    yield sse("token", {"text": token})
            else:
                async for event, data in stream_with_calculator(prompt):
                    if event == "token":
                        if first_token_ms is None:
                            first_token_ms = (perf_counter() - model_started) * 1000
                        answer_parts.append(data)
                        yield sse("token", {"text": data})
                    else:
                        if event == "tool_result":
                            tool_activity.append(data)
                        yield sse(event, data)
            answer = "".join(answer_parts).strip() or FALLBACK_ANSWER
            sources = public_sources(prompt_chunks)
            usage = getattr(gemini, "last_usage", None)
            if callable(usage):
                usage = usage()
            tokens = usage or {
                "input_tokens": estimated_tokens(prompt),
                "output_tokens": estimated_tokens(answer),
                "total_tokens": estimated_tokens(prompt) + estimated_tokens(answer),
            }
            await append_message(
                db, settings, conversation["_id"],
                {"role": "assistant", "content": answer, "sources": sources, "tool_activity": tool_activity, "created_at": now_utc()},
                answer,
            )
            emit("chat_metrics", request_id=request_id, conversation_id=conversation["_id"], provider=settings.gemini_provider, model=settings.gemini_model, requested_top_k=payload.top_k, effective_top_k=effective_top_k, retrieved_chunks=len(chunks), scores=[chunk["score"] for chunk in chunks], tool_names=[item.get("name") for item in tool_activity], retrieval_ms=round(retrieval_ms, 2), prompt_build_ms=round(prompt_build_ms, 2), model_first_token_ms=round(first_token_ms if first_token_ms is not None else (perf_counter() - model_started) * 1000, 2), model_total_ms=round((perf_counter() - model_started) * 1000, 2), request_total_ms=round((perf_counter() - request_started) * 1000, 2), **tokens, token_source="provider" if usage else "estimated", status="completed")
            yield sse("sources", sources)
            yield sse("done", {"status": "completed"})
        except Exception:
            emit("chat_metrics", request_id=request_id, conversation_id=conversation["_id"], provider=settings.gemini_provider, model=settings.gemini_model, effective_top_k=effective_top_k, request_total_ms=round((perf_counter() - request_started) * 1000, 2), status="failed", error_code="generation_failed")
            yield sse("error", {"message": "Unable to generate response"})


    return StreamingResponse(events(), media_type="text/event-stream")


app.include_router(router)
