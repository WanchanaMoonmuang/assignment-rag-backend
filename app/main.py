from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.auth import create_access_token, credentials_match, get_current_username
from app.rag import (
    FALLBACK_ANSWER,
    GeminiClient,
    build_prompt,
    chunk_text,
    clamp_top_k,
    make_id,
    public_sources,
    question_fits_budget,
    sse,
)
from app.schemas import ChatRequest, IngestRequest, LoginRequest
from app.settings import Settings, get_settings


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
    [query_embedding] = await gemini.embed_texts([question])
    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.mongodb_vector_index,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max(top_k * 20, 100),
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 1,
                "document_id": 1,
                "document_name": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    rows = await chunk_col(db, settings).aggregate(pipeline).to_list(length=top_k)
    chunks = []
    for row in rows:
        score = float(row.get("score") or 0)
        if score < settings.rag_min_score:
            continue
        content = row.get("content", "")
        chunks.append(
            {
                "document_id": row["document_id"],
                "document_name": row["document_name"],
                "chunk_id": row["_id"],
                "content": content,
                "snippet": content[:500],
                "score": score,
            }
        )
    return chunks


async def completed_answer(
    payload: ChatRequest,
    db: Any,
    settings: Settings,
    gemini: GeminiClient,
    history_messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    chunks = await retrieve_chunks(
        payload.question, clamp_top_k(payload.top_k, settings.rag_top_k), db, settings, gemini
    )
    prompt, prompt_chunks = build_prompt(
        chunks, payload.question, history_messages, settings.generation_context_token_budget
    )
    answer = await gemini.generate(prompt)
    return answer or FALLBACK_ANSWER, public_sources(prompt_chunks)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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


@router.post("/ingest")
async def ingest(
    payload: IngestRequest,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if len(payload.content.encode("utf-8")) > settings.max_document_bytes:
        raise HTTPException(status_code=413, detail="Document content exceeds 1 MB limit")

    chunks = chunk_text(payload.content, settings.rag_chunk_size, settings.rag_chunk_overlap)
    embeddings = await gemini.embed_texts(chunks)
    if len(embeddings) != len(chunks):
        raise HTTPException(status_code=502, detail="Embedding count did not match chunk count")

    metadata = payload.metadata or {}
    source = metadata.get("source", "plain_text")
    created_at = now_utc()
    document_id = make_id("doc")
    document = {
        "_id": document_id,
        "document_name": payload.document_name,
        "source": source,
        "chunks_count": len(chunks),
        "metadata": metadata,
        "created_at": created_at,
        "updated_at": created_at,
    }
    chunk_documents = [
        {
            "_id": make_id("chunk"),
            "document_id": document_id,
            "document_name": payload.document_name,
            "chunk_index": index,
            "content": chunk,
            "embedding": embedding,
            "metadata": metadata,
            "created_at": created_at,
        }
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1)
    ]

    documents = doc_col(db, settings)
    chunks_collection = chunk_col(db, settings)
    session = await db.client.start_session()
    try:
        async with session:
            async with session.start_transaction():
                await documents.insert_one(document, session=session)
                await chunks_collection.insert_many(chunk_documents, session=session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to ingest document") from exc

    return {
        "status": "success",
        "document_id": document_id,
        "document_name": payload.document_name,
        "chunks_created": len(chunks),
    }


@router.get("/documents")
async def list_documents(
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    cursor = doc_col(db, settings).find({}).sort("created_at", -1)
    documents = await cursor.to_list(length=None)
    return {"documents": [serialize_document(document) for document in documents]}


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    documents = doc_col(db, settings)
    chunks = chunk_col(db, settings)
    session = await db.client.start_session()
    try:
        async with session:
            async with session.start_transaction():
                document = await documents.find_one({"_id": document_id}, session=session)
                if document is None:
                    raise HTTPException(status_code=404, detail="Document not found")
                deleted = await chunks.delete_many({"document_id": document_id}, session=session)
                metadata_deleted = await documents.delete_one({"_id": document_id}, session=session)
                if metadata_deleted.deleted_count != 1:
                    raise RuntimeError("document metadata delete failed")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to delete document") from exc
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
    conversation, history_messages = await prepare_chat(payload, db, settings)
    answer, sources = await completed_answer(payload, db, settings, gemini, history_messages)
    await append_message(
        db,
        settings,
        conversation["_id"],
        {"role": "assistant", "content": answer, "sources": sources, "created_at": now_utc()},
        answer,
    )
    return {"conversation_id": conversation["_id"], "answer": answer, "sources": sources}


@router.post("/chat/stream")
async def stream_chat(
    payload: ChatRequest,
    _: str = Depends(get_current_username),
    db: Any = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    conversation, history_messages = await prepare_chat(payload, db, settings)

    async def events() -> AsyncIterator[str]:
        yield sse(
            "conversation",
            {"conversation_id": conversation["_id"], "title": conversation["title"]},
        )
        try:
            chunks = await retrieve_chunks(
                payload.question, clamp_top_k(payload.top_k, settings.rag_top_k), db, settings, gemini
            )
            prompt, prompt_chunks = build_prompt(
                chunks, payload.question, history_messages, settings.generation_context_token_budget
            )
            answer_parts: list[str] = []
            async for token in gemini.stream(prompt):
                answer_parts.append(token)
                yield sse("token", {"text": token})
            answer = "".join(answer_parts).strip() or FALLBACK_ANSWER
            sources = public_sources(prompt_chunks)
            yield sse("sources", sources)
            await append_message(
                db,
                settings,
                conversation["_id"],
                {"role": "assistant", "content": answer, "sources": sources, "created_at": now_utc()},
                answer,
            )
            yield sse("done", {"status": "completed"})
        except Exception:
            yield sse("error", {"message": "Unable to generate response"})

    return StreamingResponse(events(), media_type="text/event-stream")


app.include_router(router)
