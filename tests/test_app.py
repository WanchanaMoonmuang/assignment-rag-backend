from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pymongo.errors import OperationFailure

from app.main import app, configure_gemini_client
from app.rag import SYSTEM_INSTRUCTION, GeminiClient, build_prompt, chunk_text, question_fits_budget
from app.settings import Settings, get_settings
from app.worker import claim_job, process_job


class Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def sort(self, key: str, direction: int) -> "Cursor":
        self.rows.sort(key=lambda row: row.get(key), reverse=direction == -1)
        return self

    async def to_list(self, length: int | None) -> list[dict[str, Any]]:
        rows = self.rows if length is None else self.rows[:length]
        return deepcopy(rows)


class Collection:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.aggregate_rows: list[dict[str, Any]] = []
        self.last_pipeline: list[dict[str, Any]] | None = None
        self.fail_aggregate = False
        self.fail_delete_many = False
        self.fail_delete_one = False
        self.fail_insert_many = False

    async def insert_one(self, doc: dict[str, Any], session: Any = None) -> None:
        self.docs[doc["_id"]] = deepcopy(doc)

    async def insert_many(self, docs: list[dict[str, Any]], session: Any = None) -> None:
        if self.fail_insert_many:
            raise RuntimeError("insert_many failed")
        for doc in docs:
            self.docs[doc["_id"]] = deepcopy(doc)

    def find(self, query: dict[str, Any] | None = None, projection: dict[str, int] | None = None):
        rows = list(self.docs.values())
        if projection and projection.get("messages") == 0:
            rows = [{k: v for k, v in row.items() if k != "messages"} for row in rows]
        return Cursor(deepcopy(rows))

    async def find_one(self, query: dict[str, Any], session: Any = None) -> dict[str, Any] | None:
        if "_id" in query:
            doc = self.docs.get(query["_id"])
            return deepcopy(doc) if doc else None
        return None

    async def find_one_and_delete(self, query: dict[str, Any], session: Any = None) -> dict[str, Any] | None:
        doc = self.docs.pop(query["_id"], None)
        return deepcopy(doc) if doc else None

    async def delete_one(self, query: dict[str, Any], session: Any = None) -> Any:
        if self.fail_delete_one:
            raise RuntimeError("delete_one failed")
        deleted = 1 if self.docs.pop(query["_id"], None) else 0
        return SimpleNamespace(deleted_count=deleted)

    async def delete_many(self, query: dict[str, Any], session: Any = None) -> Any:
        if self.fail_delete_many:
            raise RuntimeError("delete_many failed")
        document_id = query["document_id"]
        ids = [doc_id for doc_id, doc in self.docs.items() if doc.get("document_id") == document_id]
        for doc_id in ids:
            del self.docs[doc_id]
        return SimpleNamespace(deleted_count=len(ids))

    async def update_one(self, query: dict[str, Any], update: dict[str, Any], session: Any = None) -> None:
        doc = self.docs[query["_id"]]
        for key, value in update.get("$set", {}).items():
            doc[key] = value
        for key, value in update.get("$push", {}).items():
            doc.setdefault(key, []).append(deepcopy(value))
        return SimpleNamespace(modified_count=1)

    async def update_many(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        updated = 0
        for doc in self.docs.values():
            expires_before = query.get("lease_expires_at", {}).get("$lt")
            attempts_at_least = query.get("attempts", {}).get("$gte")
            if query.get("status") != doc.get("status"):
                continue
            if expires_before and not doc.get("lease_expires_at") < expires_before:
                continue
            if attempts_at_least is not None and doc.get("attempts", 0) < attempts_at_least:
                continue
            for key, value in update.get("$set", {}).items():
                doc[key] = value
            updated += 1
        return SimpleNamespace(modified_count=updated)


    async def find_one_and_update(self, *args: Any, **kwargs: Any) -> None:
        return None


    def aggregate(self, pipeline: list[dict[str, Any]]) -> Cursor:
        self.last_pipeline = deepcopy(pipeline)
        if self.fail_aggregate:
            raise OperationFailure("search index unavailable")
        return Cursor(deepcopy(self.aggregate_rows))


class FakeSession:
    def __init__(self, db: "FakeDB") -> None:
        self.db = db
        self.snapshot: dict[str, dict[str, dict[str, Any]]] = {}

    def start_transaction(self) -> "FakeSession":
        return self

    async def __aenter__(self) -> "FakeSession":
        self.snapshot = {name: deepcopy(col.docs) for name, col in self.db.collections.items()}
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            for name, docs in self.snapshot.items():
                self.db.collections[name].docs = deepcopy(docs)
        return None


class FakeClient:
    def __init__(self, db: "FakeDB") -> None:
        self.db = db

    async def start_session(self) -> FakeSession:
        return FakeSession(self.db)


class FakeDB:
    def __init__(self) -> None:
        self.collections = {
            "documents": Collection(),
            "document_chunks": Collection(),
            "conversations": Collection(),
            "ingestion_jobs": Collection(),
        }
        self.client = FakeClient(self)

    def __getitem__(self, name: str) -> Collection:
        return self.collections[name]


class FakeGemini:
    def __init__(self) -> None:
        self.generate_prompts: list[str] = []
        self.stream_prompts: list[str] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    async def generate(self, prompt: str) -> str:
        self.generate_prompts.append(prompt)
        return "Generated answer."

    async def stream(self, prompt: str):
        self.stream_prompts.append(prompt)
        yield "Generated"
        yield " answer."


def test_embed_texts_submits_each_chunk_separately() -> None:
    calls: list[list[object]] = []

    class FakeModels:
        def embed_content(self, **kwargs: Any) -> Any:
            calls.append(kwargs["contents"])
            index = len(calls)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(index)])])

    client = GeminiClient.__new__(GeminiClient)
    client._settings = SimpleNamespace(gemini_embedding_model="test", gemini_embedding_dimensions=2)
    client._client = SimpleNamespace(models=FakeModels())
    client._types = SimpleNamespace(
        Content=lambda parts: parts,
        Part=SimpleNamespace(from_text=lambda text: text),
        EmbedContentConfig=lambda **kwargs: kwargs,
    )

    assert asyncio.run(client.embed_texts(["first", "second"])) == [[1.0], [2.0]]
    assert calls == [[["first"]], [["second"]]]


@pytest.fixture
def client() -> TestClient:
    settings = Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        app.state.db = FakeDB()
        app.state.gemini = FakeGemini()
        yield test_client
    app.dependency_overrides.clear()


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/login", json={"username": "admin", "password": "adminRAG123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def install_fake_genai(monkeypatch: pytest.MonkeyPatch, client_cls: type[Any]) -> None:
    class Part:
        @staticmethod
        def from_text(*, text: str) -> str:
            return text

    class Content:
        def __init__(self, parts: list[str]) -> None:
            self.parts = parts

    class EmbedContentConfig:
        def __init__(self, output_dimensionality: int) -> None:
            self.output_dimensionality = output_dimensionality

    class GenerateContentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        Part=Part,
        Content=Content,
        EmbedContentConfig=EmbedContentConfig,
        GenerateContentConfig=GenerateContentConfig,
    )
    fake_genai = SimpleNamespace(Client=client_cls, types=fake_types)
    monkeypatch.setitem(__import__("sys").modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(__import__("sys").modules, "google.genai", fake_genai)
    monkeypatch.setitem(__import__("sys").modules, "google.genai.types", fake_types)


def test_gemini_developer_api_uses_models_api(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls: list[dict[str, Any]] = []
    generate_calls: list[dict[str, Any]] = []
    stream_calls: list[dict[str, Any]] = []

    class Models:
        def generate_content(self, **kwargs: Any) -> Any:
            generate_calls.append(kwargs)
            return SimpleNamespace(text="generated")

        def generate_content_stream(self, **kwargs: Any) -> Any:
            stream_calls.append(kwargs)
            return iter([SimpleNamespace(text="streamed")])

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            init_calls.append(kwargs)
            self.models = Models()

    install_fake_genai(monkeypatch, Client)

    client = GeminiClient(Settings(gemini_provider="developer_api", gemini_api_key="key", gemini_temperature=0.37, _env_file=None))
    assert asyncio.run(client.generate("prompt")) == "generated"

    async def collect_stream() -> list[str]:
        return [token async for token in client.stream("prompt")]

    assert asyncio.run(collect_stream()) == ["streamed"]
    assert init_calls == [{"api_key": "key"}]
    assert generate_calls[0]["config"].kwargs["temperature"] == 0.37
    assert stream_calls[0]["config"].kwargs["temperature"] == 0.37



def test_gemini_vertex_ai_uses_project_location_and_model_api(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls: list[dict[str, Any]] = []
    generate_calls: list[dict[str, Any]] = []
    stream_calls: list[dict[str, Any]] = []
    embed_calls: list[dict[str, Any]] = []

    class Models:
        def embed_content(self, **kwargs: Any) -> Any:
            embed_calls.append(kwargs)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])])

        def generate_content(self, **kwargs: Any) -> Any:
            generate_calls.append(kwargs)
            return SimpleNamespace(text="generated")

        def generate_content_stream(self, **kwargs: Any) -> Any:
            stream_calls.append(kwargs)
            return iter([SimpleNamespace(text="streamed"), SimpleNamespace(text="")])

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            init_calls.append(kwargs)
            self.models = Models()

    install_fake_genai(monkeypatch, Client)

    client = GeminiClient(
        Settings(
            gemini_provider="vertex_ai",
            google_cloud_project="project-1",
            google_cloud_location="asia-southeast1",
            gemini_temperature=0.63,
        )
    )
    assert asyncio.run(client.embed_texts(["first", "second"])) == [[0.1], [0.1]]
    assert [[content.parts for content in call["contents"]] for call in embed_calls] == [[["first"]], [["second"]]]
    assert asyncio.run(client.generate("prompt")) == "generated"

    async def collect_stream() -> list[str]:
        return [token async for token in client.stream("prompt")]

    assert asyncio.run(collect_stream()) == ["streamed"]
    assert init_calls == [
        {"vertexai": True, "project": "project-1", "location": "asia-southeast1"}
    ]
    assert generate_calls[0]["contents"] == "prompt"
    assert stream_calls[0]["contents"] == "prompt"
    assert generate_calls[0]["config"].kwargs["temperature"] == 0.63
    assert stream_calls[0]["config"].kwargs["temperature"] == 0.63


def test_gemini_client_requires_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

    install_fake_genai(monkeypatch, Client)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient(Settings(gemini_provider="developer_api", gemini_api_key=None, _env_file=None))

    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        GeminiClient(
            Settings(
                gemini_provider="vertex_ai",
                google_cloud_project=None,
                gcp_project_id=None,
                _env_file=None,
            )
        )

    with pytest.raises(RuntimeError, match="GEMINI_PROVIDER"):
        GeminiClient(Settings(gemini_provider="unknown", gemini_api_key="key", _env_file=None))


def test_chunk_text_overlaps() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_history_context_window_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HISTORY_CONTEXT_WINDOW", "3")
    assert Settings().history_context_window == 3


def test_vertex_ai_settings_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_PROVIDER", "vertex_ai")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-1")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")

    settings = Settings()

    assert settings.gemini_provider == "vertex_ai"
    assert settings.google_cloud_project == "project-1"
    assert settings.google_cloud_location == "asia-southeast1"


def test_vertex_ai_accepts_gcp_project_id_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            init_calls.append(kwargs)
            self.models = SimpleNamespace()

    install_fake_genai(monkeypatch, Client)

    GeminiClient(
        Settings(
            gemini_provider="vertex_ai",
            google_cloud_project=None,
            gcp_project_id="fallback-project",
            google_cloud_location="us-central1",
            _env_file=None,
        )
    )

    assert init_calls == [
        {"vertexai": True, "project": "fallback-project", "location": "us-central1"}
    ]


def test_vertex_ai_gcp_project_id_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "fallback-project")

    settings = Settings(_env_file=None)

    assert settings.google_cloud_project is None
    assert settings.gcp_project_id == "fallback-project"


def test_auth_flow(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.post("/api/auth/login", json={"username": "admin", "password": "bad"}).status_code == 401
    response = client.get("/api/auth/me", headers=auth_headers(client))
    assert response.status_code == 200
    assert response.json() == {"username": "admin"}


def test_auth_me_fails_closed_without_jwt_secret(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="",
    )
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer anything"})
    assert response.status_code == 500


def test_chat_reports_gemini_startup_config_error(client: TestClient) -> None:
    app.state.gemini = None
    app.state.gemini_error = "GOOGLE_CLOUD_PROJECT is not configured"

    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"question": "What is the refund policy?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "GOOGLE_CLOUD_PROJECT is not configured"


def test_configure_gemini_client_clears_stale_client_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_client = object()
    app.state.gemini = stale_client

    class FailingGeminiClient:
        def __init__(self, settings: Settings) -> None:
            raise RuntimeError("GEMINI_API_KEY is not configured")

    monkeypatch.setattr("app.main.GeminiClient", FailingGeminiClient)

    configure_gemini_client(app, Settings(gemini_api_key=None, _env_file=None))

    assert app.state.gemini is None
    assert app.state.gemini_error == "GEMINI_API_KEY is not configured"

def test_ingest_lists_and_deletes_document(client: TestClient) -> None:
    headers = auth_headers(client)
    response = client.post(
        "/api/ingest",
        headers=headers,
        json={"document_name": "policy.txt", "content": "Refunds are available."},
    )
    assert response.status_code == 202
    payload = response.json()
    job = app.state.db["ingestion_jobs"].docs[payload["job_id"]]
    job.update({"status": "processing", "lease_token": "test-lease", "attempts": 1})
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))

    list_response = client.get("/api/documents", headers=headers)
    assert list_response.json()["documents"][0]["document_id"] == payload["document_id"]

    delete_response = client.delete(f"/api/documents/{payload['document_id']}", headers=headers)
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted_chunks"] == 1


def test_ingest_rolls_back_if_chunk_insert_fails(client: TestClient) -> None:
    headers = auth_headers(client)
    app.state.db["document_chunks"].fail_insert_many = True
    response = client.post(
        "/api/ingest",
        headers=headers,
        json={"document_name": "policy.txt", "content": "Refunds are available."},
    )
    assert response.status_code == 202
    job = app.state.db["ingestion_jobs"].docs[response.json()["job_id"]]
    job.update({"status": "processing", "lease_token": "test-lease", "attempts": 1})
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))

    assert app.state.db["ingestion_jobs"].docs[response.json()["job_id"]]["status"] == "queued"
    assert app.state.db["documents"].docs == {}
    assert app.state.db["document_chunks"].docs == {}


def test_document_delete_rolls_back_if_chunk_delete_fails(client: TestClient) -> None:
    headers = auth_headers(client)
    response = client.post(
        "/api/ingest",
        headers=headers,
        json={"document_name": "policy.txt", "content": "Refunds are available."},
    )
    payload = response.json()
    job = app.state.db["ingestion_jobs"].docs[payload["job_id"]]
    job.update({"status": "processing", "lease_token": "test-lease", "attempts": 1})
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))
    app.state.db["document_chunks"].fail_delete_many = True

    delete_response = client.delete(f"/api/documents/{payload['document_id']}", headers=headers)
    assert delete_response.status_code == 500
    assert payload["document_id"] in app.state.db["documents"].docs


def test_document_delete_rolls_back_if_metadata_delete_fails(client: TestClient) -> None:
    headers = auth_headers(client)
    response = client.post(
        "/api/ingest",
        headers=headers,
        json={"document_name": "policy.txt", "content": "Refunds are available."},
    )
    payload = response.json()
    job = app.state.db["ingestion_jobs"].docs[payload["job_id"]]
    job.update({"status": "processing", "lease_token": "test-lease", "attempts": 1})
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))
    app.state.db["documents"].fail_delete_one = True

    delete_response = client.delete(f"/api/documents/{payload['document_id']}", headers=headers)
    assert delete_response.status_code == 500
    assert payload["document_id"] in app.state.db["documents"].docs


def test_chat_without_retrieval_uses_foundation_answer(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"question": "What is the refund policy?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Generated answer."
    assert body["sources"] == []
    assert body["conversation_id"].startswith("conv_")


def test_chat_uses_env_default_top_k(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
        rag_top_k=7,
    )
    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"question": "What is the refund policy?"},
    )
    assert response.status_code == 200
    rank_fusion = next(iter(app.state.db["document_chunks"].last_pipeline[0].values()))
    pipelines = rank_fusion["input"]["pipelines"]
    vector_stage = next(iter(pipelines["vector"][0].values()))
    assert vector_stage["limit"] == 28
    lexical_stage = next(iter(pipelines["lexical"][0].values()))
    assert lexical_stage["text"]["path"] == "content"
    assert list(app.state.db["document_chunks"].last_pipeline[1].values()) == [7]


def test_chat_returns_sources_when_context_matches(client: TestClient) -> None:
    app.state.db["document_chunks"].aggregate_rows = [
        {
            "_id": "chunk_1",
            "document_id": "doc_1",
            "document_name": "policy.txt",
            "content": "Refunds are available within 30 days.",
            "score": 0.9,
        }
    ]
    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"question": "What is the refund policy?", "top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Generated answer."
    assert body["sources"][0]["chunk_id"] == "chunk_1"


def test_chat_prompt_includes_prior_conversation_history(client: TestClient) -> None:
    app.state.db["document_chunks"].aggregate_rows = [
        {
            "_id": "chunk_1",
            "document_id": "doc_1",
            "document_name": "policy.txt",
            "content": "Refunds are available within 30 days.",
            "score": 0.9,
        }
    ]
    app.state.db["conversations"].docs["conv_1"] = {
        "_id": "conv_1",
        "title": "Refunds",
        "messages": [
            {"role": "user", "content": "I bought a product yesterday."},
            {"role": "assistant", "content": "I can help with policy questions.", "sources": []},
        ],
    }

    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"conversation_id": "conv_1", "question": "What is the refund policy?", "top_k": 5},
    )

    assert response.status_code == 200
    prompt = app.state.gemini.generate_prompts[-1]
    assert "Chat history:" in prompt
    assert "user: I bought a product yesterday." in prompt
    assert "assistant: I can help with policy questions." in prompt
    assert prompt.count("What is the refund policy?") == 1
    assert "Question:\nWhat is the refund policy?" in prompt


def test_stream_chat_prompt_includes_prior_conversation_history(client: TestClient) -> None:
    app.state.db["document_chunks"].aggregate_rows = [
        {
            "_id": "chunk_1",
            "document_id": "doc_1",
            "document_name": "policy.txt",
            "content": "Escalation code is ESC-42.",
            "score": 0.9,
        }
    ]
    app.state.db["conversations"].docs["conv_1"] = {
        "_id": "conv_1",
        "title": "Escalation",
        "messages": [
            {"role": "user", "content": "Remember I need the escalation code."},
            {"role": "assistant", "content": "The document mentions ESC-42.", "sources": []},
        ],
    }

    response = client.post(
        "/api/chat/stream",
        headers=auth_headers(client),
        json={"conversation_id": "conv_1", "question": "Repeat it.", "top_k": 5},
    )

    assert response.status_code == 200
    prompt = app.state.gemini.stream_prompts[-1]
    assert "Chat history:" in prompt
    assert "user: Remember I need the escalation code." in prompt
    assert "assistant: The document mentions ESC-42." in prompt
    assert "Question:\nRepeat it." in prompt


def test_chat_prompt_respects_history_context_window(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
        history_context_window=1,
    )
    app.state.db["document_chunks"].aggregate_rows = [
        {
            "_id": "chunk_1",
            "document_id": "doc_1",
            "document_name": "policy.txt",
            "content": "Refunds are available within 30 days.",
            "score": 0.9,
        }
    ]
    app.state.db["conversations"].docs["conv_1"] = {
        "_id": "conv_1",
        "title": "Refunds",
        "messages": [
            {"role": "user", "content": "Oldest message"},
            {"role": "assistant", "content": "Latest message", "sources": []},
        ],
    }

    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"conversation_id": "conv_1", "question": "What is the refund policy?", "top_k": 5},
    )

    assert response.status_code == 200
    prompt = app.state.gemini.generate_prompts[-1]
    assert "Oldest message" not in prompt
    assert "assistant: Latest message" in prompt


def test_chat_prompt_omits_history_when_window_is_zero(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
        history_context_window=0,
    )
    app.state.db["document_chunks"].aggregate_rows = [
        {
            "_id": "chunk_1",
            "document_id": "doc_1",
            "document_name": "policy.txt",
            "content": "Refunds are available within 30 days.",
            "score": 0.9,
        }
    ]
    app.state.db["conversations"].docs["conv_1"] = {
        "_id": "conv_1",
        "title": "Refunds",
        "messages": [{"role": "user", "content": "Prior message"}],
    }

    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"conversation_id": "conv_1", "question": "What is the refund policy?", "top_k": 5},
    )

    assert response.status_code == 200
    prompt = app.state.gemini.generate_prompts[-1]
    assert "Chat history:" not in prompt
    assert "Prior message" not in prompt
    assert "Question:\nWhat is the refund policy?" in prompt


def test_stream_chat_event_order(client: TestClient) -> None:
    app.state.db["document_chunks"].aggregate_rows = [
        {
            "_id": "chunk_1",
            "document_id": "doc_1",
            "document_name": "policy.txt",
            "content": "Refunds are available within 30 days.",
            "score": 0.9,
        }
    ]
    response = client.post(
        "/api/chat/stream",
        headers=auth_headers(client),
        json={"question": "What is the refund policy?"},
    )
    assert response.status_code == 200
    text = response.text
    assert "event: conversation" in text
    assert "event: token" in text
    assert "event: sources" in text
    assert "event: done" in text

def test_chat_top_k_zero_skips_retrieval(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"question": "What is the refund policy?", "top_k": 0},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Generated answer."
    assert response.json()["sources"] == []
    assert app.state.db["document_chunks"].last_pipeline is None


def test_prompt_budget_discards_chunks_before_history() -> None:
    chunks = [
        {
            "document_name": "first.txt",
            "chunk_id": "chunk_1",
            "content": "x" * 400,
        },
        {
            "document_name": "second.txt",
            "chunk_id": "chunk_2",
            "content": "y" * 400,
        },
    ]
    history = [
        {"role": "user", "content": "Earlier request"},
        {"role": "assistant", "content": "Latest answer"},
    ]

    prompt, retained_chunks = build_prompt(chunks, "Current question", history, token_budget=100)

    assert retained_chunks == []
    assert "Earlier request" in prompt
    assert "Latest answer" in prompt


def test_prompt_policy_requires_language_and_attribution() -> None:
    assert "language of the user's latest question" in SYSTEM_INSTRUCTION
    assert "inline markers" in SYSTEM_INSTRUCTION
    assert "General knowledge" in SYSTEM_INSTRUCTION
    assert "Do not claim that uncited general knowledge came from a document" in SYSTEM_INSTRUCTION


def test_question_budget_is_unicode_safe() -> None:
    assert not question_fits_budget("\U0001F600" * 30, token_budget=100)


def test_chat_rejects_question_over_context_budget(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
        generation_context_token_budget=100,
    )

    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"question": "\U0001F600" * 30},
    )

    assert response.status_code == 422
    assert app.state.db["conversations"].docs == {}


def test_text_ingestion_enqueues_and_worker_publishes_document(client: TestClient) -> None:
    response = client.post(
        "/api/ingestions/text",
        headers=auth_headers(client),
        json={
            "document_name": "policy.txt",
            "content": "Refunds are available within 30 days.",
            "metadata": {"department": "support"},
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"

    status_response = client.get(
        f"/api/ingestions/{payload['job_id']}",
        headers=auth_headers(client),
    )
    assert status_response.status_code == 200
    job = app.state.db["ingestion_jobs"].docs[payload["job_id"]]
    job.update({"status": "processing", "lease_token": "test-lease", "attempts": 1})
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))

    assert app.state.db["ingestion_jobs"].docs[payload["job_id"]]["status"] == "completed"
    assert payload["document_id"] in app.state.db["documents"].docs


def test_file_ingestion_rejects_oversized_file_before_upload(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
        max_upload_bytes=3,
    )

    response = client.post(
        "/api/ingestions/file",
        headers=auth_headers(client),
        files={"file": ("policy.txt", b"four", "text/plain")},
    )

    assert response.status_code == 413
    assert app.state.db["ingestion_jobs"].docs == {}


def test_file_ingestion_returns_failed_job_when_original_upload_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_upload(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr("app.main.upload_object", fail_upload)
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
        gcs_bucket_name="test-bucket",
    )

    response = client.post(
        "/api/ingestions/file",
        headers=auth_headers(client),
        files={"file": ("policy.txt", b"content", "text/plain")},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "failed"
    assert response.json()["error"]["code"] == "original_upload_failed"


def test_text_ingestion_counts_raw_whitespace_bytes(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
        max_upload_bytes=3,
    )

    response = client.post(
        "/api/ingestions/text",
        headers=auth_headers(client),
        json={"document_name": "policy.txt", "content": "   x"},
    )

    assert response.status_code == 413
    assert app.state.db["ingestion_jobs"].docs == {}


def test_file_ingestion_rejects_mismatched_signature_before_job(client: TestClient) -> None:
    response = client.post(
        "/api/ingestions/file",
        headers=auth_headers(client),
        files={"file": ("report.pdf", b"not a PDF", "application/pdf")},
    )

    assert response.status_code == 415
    assert app.state.db["ingestion_jobs"].docs == {}


def test_claim_marks_expired_final_attempt_as_failed(client: TestClient) -> None:
    settings = app.dependency_overrides[get_settings]()
    job_id = "job_expired"
    app.state.db["ingestion_jobs"].docs[job_id] = {
        "_id": job_id,
        "status": "processing",
        "attempts": settings.ingestion_job_max_attempts,
        "lease_expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
    }

    assert asyncio.run(claim_job(app.state.db, settings, "worker")) is None
    assert app.state.db["ingestion_jobs"].docs[job_id]["status"] == "failed"


def test_stale_lease_cannot_publish_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = client.post(
        "/api/ingestions/text",
        headers=auth_headers(client),
        json={"document_name": "policy.txt", "content": "Refunds are available."},
    )
    job = app.state.db["ingestion_jobs"].docs[response.json()["job_id"]]
    job.update({"status": "processing", "lease_token": "stale", "attempts": 1})

    async def reject_fenced_update(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(modified_count=0)

    monkeypatch.setattr(app.state.db["ingestion_jobs"], "update_one", reject_fenced_update)
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))

    assert app.state.db["documents"].docs == {}
    assert app.state.db["document_chunks"].docs == {}


def test_worker_records_processing_timeout(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    response = client.post(
        "/api/ingestions/text",
        headers=auth_headers(client),
        json={"document_name": "policy.txt", "content": "Refunds are available."},
    )
    job = app.state.db["ingestion_jobs"].docs[response.json()["job_id"]]
    job.update({"status": "processing", "lease_token": "test-lease", "attempts": 1})

    async def timeout(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError

    monkeypatch.setattr("app.worker.process_text_job", timeout)
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))

    stored = app.state.db["ingestion_jobs"].docs[job["_id"]]
    assert stored["status"] == "queued"
    assert stored["error"]["code"] == "processing_timeout"



def test_file_worker_downloads_extracts_and_publishes_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def upload(*args: Any, **kwargs: Any) -> None:
        return None

    async def download(*args: Any, **kwargs: Any) -> bytes:
        return b"First line.\nSecond line."

    monkeypatch.setattr("app.main.upload_object", upload)
    monkeypatch.setattr("app.worker.download_object", download)
    response = client.post(
        "/api/ingestions/file",
        headers=auth_headers(client),
        files={"file": ("policy.txt", b"First line.\nSecond line.", "text/plain")},
        data={"metadata_json": '{"department":"support"}'},
    )

    assert response.status_code == 202
    payload = response.json()
    job = app.state.db["ingestion_jobs"].docs[payload["job_id"]]
    job.update({"status": "processing", "lease_token": "test-lease", "attempts": 1})
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))

    document = app.state.db["documents"].docs[payload["document_id"]]
    chunk = next(iter(app.state.db["document_chunks"].docs.values()))
    assert document["source_format"] == "txt"
    assert document["original_object_name"] == job["object_name"]
    assert document["metadata"] == {"department": "support"}
    assert chunk["location"]["label"] == "Lines 1-2"
    assert chunk["chunk_type"] == "prose"



def test_expired_final_file_job_schedules_cleanup(client: TestClient) -> None:
    settings = app.dependency_overrides[get_settings]()
    app.state.db["ingestion_jobs"].docs["job_expired_file"] = {
        "_id": "job_expired_file",
        "document_id": "doc_expired_file",
        "document_name": "policy.txt",
        "source_kind": "file",
        "object_name": "originals/doc_expired_file/policy.txt",
        "status": "processing",
        "attempts": settings.ingestion_job_max_attempts,
        "lease_expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
    }

    assert asyncio.run(claim_job(app.state.db, settings, "worker")) is None
    cleanup_jobs = [
        job
        for job in app.state.db["ingestion_jobs"].docs.values()
        if job["status"] == "cleanup_pending"
    ]
    assert len(cleanup_jobs) == 1
    assert cleanup_jobs[0]["object_name"] == "originals/doc_expired_file/policy.txt"


def test_terminal_file_failure_records_cleanup_atomically(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def upload(*args: Any, **kwargs: Any) -> None:
        return None

    async def download(*args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr("app.main.upload_object", upload)
    monkeypatch.setattr("app.worker.download_object", download)
    app.dependency_overrides[get_settings] = lambda: Settings(
        auth_password="adminRAG123",
        jwt_secret_key="test-secret-key-with-at-least-32-bytes",
        mongodb_vector_index="test-index",
        ingestion_job_max_attempts=1,
    )
    response = client.post(
        "/api/ingestions/file",
        headers=auth_headers(client),
        files={"file": ("policy.txt", b"content", "text/plain")},
    )
    job = app.state.db["ingestion_jobs"].docs[response.json()["job_id"]]
    job.update({"status": "processing", "lease_token": "test-lease", "attempts": 1})
    asyncio.run(process_job(app.state.db, app.dependency_overrides[get_settings](), app.state.gemini, job))

    assert job["status"] == "failed"
    assert any(
        item["status"] == "cleanup_pending"
        for item in app.state.db["ingestion_jobs"].docs.values()
    )

def test_config_requires_auth_and_returns_runtime_limits(client: TestClient) -> None:
    assert client.get("/api/config").status_code == 401

    response = client.get("/api/config", headers=auth_headers(client))

    assert response.status_code == 200
    assert response.json() == {
        "rag_top_k": {"default": 5, "min": 0, "max": 20},
        "max_upload_bytes": 20 * 1024 * 1024,
        "supported_file_extensions": ["txt", "pdf", "docx", "csv", "json"],
    }


def test_chat_returns_503_when_search_index_is_unavailable(client: TestClient) -> None:
    app.state.db["document_chunks"].fail_aggregate = True

    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"question": "What is the refund policy?", "top_k": 1},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "MongoDB search indexes are unavailable"

def test_chat_returns_source_metadata(client: TestClient) -> None:
    app.state.db["document_chunks"].aggregate_rows = [{
        "_id": "chunk_1",
        "document_id": "doc_1",
        "document_name": "policy.pdf",
        "content": "Refunds are available within 30 days.",
        "score": 0.02,
        "source_format": "pdf",
        "chunk_type": "prose",
        "location": {"page": 2},
        "metadata": {"department": "support"},
    }]

    response = client.post(
        "/api/chat",
        headers=auth_headers(client),
        json={"question": "What is the refund policy?", "top_k": 1},
    )

    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["source_format"] == "pdf"
    assert source["chunk_type"] == "prose"
    assert source["location"] == {"page": 2}
    assert source["metadata"] == {"department": "support"}

def test_calculator_supports_safe_math_and_rejects_code() -> None:
    from app.calculator import CalculatorError, calculate

    assert calculate("sqrt(9) + 2**3") == 11
    assert calculate("round(pi, 2)") == 3.14
    with pytest.raises(CalculatorError):
        calculate("__import__('os')")


def test_get_cited_chunk_returns_neighbor_window(client: TestClient) -> None:
    chunks = app.state.db["document_chunks"].docs
    for index in range(5):
        chunks[f"chunk_{index}"] = {
            "_id": f"chunk_{index}", "document_id": "doc_1", "content": str(index),
            "chunk_index": index, "location": {"page": 1}, "metadata": {},
        }

    response = client.get("/api/documents/doc_1/chunks/chunk_2", headers=auth_headers(client))

    assert response.status_code == 200
    assert [item["chunk_id"] for item in response.json()["neighbors"]] == ["chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4"]


def test_get_original_file_returns_private_gcs_bytes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def download(*args: Any, **kwargs: Any) -> bytes:
        return b"document bytes"

    monkeypatch.setattr("app.main.download_object", download)
    app.state.db["documents"].docs["doc_1"] = {
        "_id": "doc_1", "document_name": "policy.txt", "original_object_name": "originals/doc_1/policy.txt",
        "content_type": "text/plain",
    }

    response = client.get("/api/documents/doc_1/file", headers=auth_headers(client))

    assert response.status_code == 200
    assert response.content == b"document bytes"
    assert response.headers["content-type"].startswith("text/plain")
    assert "policy.txt" in response.headers["content-disposition"]

def test_stream_with_calculator_handles_multiple_calls() -> None:
    calls = [
        SimpleNamespace(name="calculator", args={"expression": "1 + 1"}),
        SimpleNamespace(name="calculator", args={"expression": "2 + 2"}),
    ]
    model_calls: list[dict[str, Any]] = []

    class Models:
        def generate_content_stream(self, **kwargs: Any) -> Any:
            model_calls.append(kwargs)
            if len(model_calls) == 1:
                return iter([SimpleNamespace(text=None, function_calls=calls)])
            return iter([SimpleNamespace(text="done", function_calls=[])])

    class Part:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        @staticmethod
        def from_function_response(**kwargs: Any) -> dict[str, Any]:
            return kwargs

    client = GeminiClient.__new__(GeminiClient)
    client._provider = "developer_api"
    client._settings = SimpleNamespace(gemini_model="test", gemini_temperature=0.2)
    client._client = SimpleNamespace(models=Models())
    client._types = SimpleNamespace(
        FunctionDeclaration=lambda **kwargs: kwargs,
        Tool=lambda **kwargs: kwargs,
        GenerateContentConfig=lambda **kwargs: kwargs,
        Part=Part,
        Content=lambda **kwargs: kwargs,
    )

    async def collect() -> list[tuple[str, Any]]:
        return [item async for item in client.stream_with_calculator("calculate")]

    events = asyncio.run(collect())
    assert [event for event, _ in events] == ["tool_call", "tool_result", "tool_call", "tool_result", "token"]
    assert events[0][1] == {"name": "calculator", "status": "requested"}
    assert events[1][1] == {"name": "calculator", "status": "completed", "display_value": "2"}
    assert len(model_calls) == 2
    assert len(model_calls[1]["contents"][1]["parts"]) == 2
    assert len(model_calls[1]["contents"][2]["parts"]) == 2
