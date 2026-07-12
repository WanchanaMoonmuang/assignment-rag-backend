from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator
from uuid import uuid4

from app.settings import Settings

FALLBACK_ANSWER = "The information is not available in the provided documents."
SYSTEM_INSTRUCTION = (
    "Answer the user's question using only the provided context. "
    f'If the answer cannot be found in the context, say: "{FALLBACK_ANSWER}" '
    "Do not make up information. Do not use outside knowledge. Keep the answer clear and concise."
)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def clamp_top_k(value: int | None, default: int) -> int:
    return min(max(value if value is not None else default, 3), 10)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def format_chat_history(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role") or "message").strip()
        content = str(message.get("content") or "").strip()
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_prompt(
    chunks: list[dict[str, Any]],
    question: str,
    history_messages: list[dict[str, Any]] | None = None,
) -> str:
    context = "\n\n".join(
        f"Source {index}: {chunk['document_name']} / {chunk['chunk_id']}\n{chunk['content']}"
        for index, chunk in enumerate(chunks, start=1)
    )
    sections = []
    history = format_chat_history(history_messages or [])
    if history:
        sections.append(f"Chat history:\n{history}")
    sections.append(f"Context:\n{context}")
    sections.append(f"Question:\n{question}")
    return "\n\n".join(sections)


def public_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("document_id", "document_name", "chunk_id", "snippet", "score")
    return [{key: chunk[key] for key in keys} for chunk in chunks]


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _next_or_none(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return None


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        from google import genai
        from google.genai import types

        self._settings = settings
        self._provider = settings.gemini_provider.lower()
        if self._provider == "developer_api":
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            self._client = genai.Client(api_key=settings.gemini_api_key)
        elif self._provider == "vertex_ai":
            project = settings.google_cloud_project or settings.gcp_project_id
            if not project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured")
            if not settings.google_cloud_location:
                raise RuntimeError("GOOGLE_CLOUD_LOCATION is not configured")
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=settings.google_cloud_location,
            )
        else:
            raise RuntimeError("GEMINI_PROVIDER must be developer_api or vertex_ai")
        self._types = types

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for text in texts:
            content = self._types.Content(parts=[self._types.Part.from_text(text=text)])
            result = await asyncio.to_thread(
                self._client.models.embed_content,
                model=self._settings.gemini_embedding_model,
                contents=[content],
                config=self._types.EmbedContentConfig(
                    output_dimensionality=self._settings.gemini_embedding_dimensions
                ),
            )
            embeddings.extend(
                [float(value) for value in embedding.values] for embedding in result.embeddings
            )
        return embeddings

    def _generate_content_config(self) -> Any:
        return self._types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        )

    async def generate(self, prompt: str) -> str:
        if self._provider == "vertex_ai":
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._settings.gemini_model,
                contents=prompt,
                config=self._generate_content_config(),
            )
            return (getattr(response, "text", "") or "").strip()

        interaction = await asyncio.to_thread(
            self._client.interactions.create,
            model=self._settings.gemini_model,
            input=prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={"temperature": 0.2},
            store=False,
        )
        return (getattr(interaction, "output_text", "") or "").strip()

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        if self._provider == "vertex_ai":
            stream = await asyncio.to_thread(
                self._client.models.generate_content_stream,
                model=self._settings.gemini_model,
                contents=prompt,
                config=self._generate_content_config(),
            )
            while True:
                chunk = await asyncio.to_thread(_next_or_none, iter(stream))
                if chunk is None:
                    break
                if getattr(chunk, "text", None):
                    yield chunk.text
            return

        stream = await asyncio.to_thread(
            self._client.interactions.create,
            model=self._settings.gemini_model,
            input=prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={"temperature": 0.2},
            stream=True,
            store=False,
        )
        while True:
            event = await asyncio.to_thread(_next_or_none, iter(stream))
            if event is None:
                break
            if getattr(event, "event_type", None) != "step.delta":
                continue
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", None) == "text" and getattr(delta, "text", None):
                yield delta.text
