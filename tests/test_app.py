from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, configure_gemini_client
from app.rag import SYSTEM_INSTRUCTION, GeminiClient, build_prompt, chunk_text, question_fits_budget
from app.settings import Settings, get_settings


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

    def aggregate(self, pipeline: list[dict[str, Any]]) -> Cursor:
        self.last_pipeline = deepcopy(pipeline)
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
        rag_min_score=0.5,
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


def test_gemini_developer_api_calls_disable_prompt_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    class Interactions:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            if kwargs.get("stream"):
                event = SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="text", text="streamed"),
                )
                return iter([event])
            return SimpleNamespace(output_text="generated")

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            init_calls.append(kwargs)
            self.interactions = Interactions()
            self.models = SimpleNamespace()

    install_fake_genai(monkeypatch, Client)

    client = GeminiClient(Settings(gemini_provider="developer_api", gemini_api_key="key", gemini_temperature=0.37, _env_file=None))
    assert asyncio.run(client.generate("prompt")) == "generated"

    async def collect_stream() -> list[str]:
        return [token async for token in client.stream("prompt")]

    assert asyncio.run(collect_stream()) == ["streamed"]
    assert init_calls == [{"api_key": "key"}]
    assert calls[0]["store"] is False
    assert calls[1]["store"] is False
    assert calls[0]["generation_config"]["temperature"] == 0.37
    assert calls[1]["generation_config"]["temperature"] == 0.37


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
        GeminiClient(Settings(gemini_api_key=None, _env_file=None))

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
        json={
            "document_name": "policy.txt",
            "content": "Refunds are available within 30 days.",
            "metadata": {"source": "txt_upload"},
        },
    )
    assert response.status_code == 200
    document_id = response.json()["document_id"]

    list_response = client.get("/api/documents", headers=headers)
    assert list_response.json()["documents"][0]["document_id"] == document_id

    delete_response = client.delete(f"/api/documents/{document_id}", headers=headers)
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
    assert response.status_code == 500
    assert app.state.db["documents"].docs == {}
    assert app.state.db["document_chunks"].docs == {}


def test_document_delete_rolls_back_if_chunk_delete_fails(client: TestClient) -> None:
    headers = auth_headers(client)
    response = client.post(
        "/api/ingest",
        headers=headers,
        json={"document_name": "policy.txt", "content": "Refunds are available."},
    )
    document_id = response.json()["document_id"]
    app.state.db["document_chunks"].fail_delete_many = True

    delete_response = client.delete(f"/api/documents/{document_id}", headers=headers)
    assert delete_response.status_code == 500

    list_response = client.get("/api/documents", headers=headers)
    assert list_response.json()["documents"][0]["document_id"] == document_id
    assert len(app.state.db["document_chunks"].docs) == 1


def test_document_delete_rolls_back_if_metadata_delete_fails(client: TestClient) -> None:
    headers = auth_headers(client)
    response = client.post(
        "/api/ingest",
        headers=headers,
        json={"document_name": "policy.txt", "content": "Refunds are available."},
    )
    document_id = response.json()["document_id"]
    app.state.db["documents"].fail_delete_one = True

    delete_response = client.delete(f"/api/documents/{document_id}", headers=headers)
    assert delete_response.status_code == 500

    list_response = client.get("/api/documents", headers=headers)
    assert list_response.json()["documents"][0]["document_id"] == document_id
    assert len(app.state.db["document_chunks"].docs) == 1


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
    vector_stage = app.state.db["document_chunks"].last_pipeline[0]["$vectorSearch"]
    assert vector_stage["limit"] == 7


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
        rag_min_score=0.5,
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
        rag_min_score=0.5,
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
