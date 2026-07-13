from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from typing import Any, AsyncIterator
from uuid import uuid4
from app.calculator import CalculatorError, calculate

from app.settings import Settings

FALLBACK_ANSWER = "I could not generate an answer. Please try again."
SYSTEM_INSTRUCTION = (
    "Answer in the language of the user's latest question. "
    "When document context is provided, support document-based claims with inline markers "
    "such as [1] that match the numbered sources. "
    "You may use general knowledge when document context is incomplete, but put those claims "
    "under a 'General knowledge' heading without source markers. "
    "Do not claim that uncited general knowledge came from a document. Keep the answer clear and concise."
)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def clamp_top_k(value: int | None, default: int) -> int:
    return min(max(value if value is not None else default, 0), 20)


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


def _prompt_text(
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


def estimated_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")))



def provider_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    if total_tokens <= 0:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

def question_fits_budget(question: str, token_budget: int) -> bool:
    return estimated_tokens(_prompt_text([], question)) <= token_budget


def build_prompt(
    chunks: list[dict[str, Any]],
    question: str,
    history_messages: list[dict[str, Any]] | None,
    token_budget: int,
) -> tuple[str, list[dict[str, Any]]]:
    if not question_fits_budget(question, token_budget):
        raise ValueError("Question exceeds the configured context budget")
    selected_chunks = list(chunks)
    selected_history = list(history_messages or [])
    while True:
        prompt = _prompt_text(selected_chunks, question, selected_history)
        if estimated_tokens(prompt) <= token_budget or (
            not selected_chunks and not selected_history
        ):
            return prompt, selected_chunks
        if selected_chunks:
            selected_chunks.pop()
        else:
            selected_history.pop(0)


def public_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "document_id", "document_name", "chunk_id", "snippet", "score",
        "source_format", "chunk_type", "location", "metadata",
    )
    return [{key: chunk[key] for key in keys if key in chunk} for chunk in chunks]


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
        self._last_usage: ContextVar[dict[str, int] | None] = ContextVar("last_usage", default=None)

    def _usage_context(self) -> ContextVar[dict[str, int] | None]:
        context = getattr(self, "_last_usage", None)
        if context is None:
            context = ContextVar("last_usage", default=None)
            self._last_usage = context
        return context


    def last_usage(self) -> dict[str, int] | None:
        return self._usage_context().get()

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
            temperature=self._settings.gemini_temperature,
        )

    async def generate(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._settings.gemini_model,
            contents=prompt,
            config=self._generate_content_config(),
        )
        return (getattr(response, "text", "") or "").strip()


    async def generate_with_calculator(
        self, prompt: str
    ) -> tuple[str, list[dict[str, Any]], dict[str, int] | None]:
        activity: list[dict[str, Any]] = []

        def calculator(expression: str) -> dict[str, Any]:
            try:
                result = calculate(expression)
                record = {"name": "calculator", "arguments": {"expression": expression}, "result": result}
            except CalculatorError as exc:
                record = {"name": "calculator", "arguments": {"expression": expression}, "error": str(exc)}
            activity.append(record)
            return record

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._settings.gemini_model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self._settings.gemini_temperature,
                tools=[calculator],
            ),
        )
        return (getattr(response, "text", "") or "").strip(), activity, provider_usage(response)


    async def stream_with_calculator(self, prompt: str) -> AsyncIterator[tuple[str, Any]]:
        declaration = self._types.FunctionDeclaration(
            name="calculator",
            description="Evaluate a mathematical expression.",
            parametersJsonSchema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
        config = self._types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self._settings.gemini_temperature,
            tools=[self._types.Tool(functionDeclarations=[declaration])],
        )
        self._usage_context().set(None)
        stream = await asyncio.to_thread(
            self._client.models.generate_content_stream,
            model=self._settings.gemini_model,
            contents=prompt,
            config=config,
        )
        function_calls: list[Any] = []
        while True:
            chunk = await asyncio.to_thread(_next_or_none, iter(stream))
            if chunk is None:
                break
            usage = provider_usage(chunk)
            if usage is not None:
                self._usage_context().set(usage)
            if getattr(chunk, "text", None):
                yield "token", chunk.text
            function_calls.extend(getattr(chunk, "function_calls", None) or [])
        if not function_calls:
            return
        response_parts = []
        call_parts = []
        for function_call in function_calls:
            arguments = dict(function_call.args or {})
            yield "tool_call", {"name": function_call.name, "status": "requested"}
            try:
                result = calculate(str(arguments.get("expression", "")))
                response_data = {"result": result}
                yield "tool_result", {
                    "name": function_call.name,
                    "status": "completed",
                    "display_value": str(result),
                }
            except CalculatorError as exc:
                response_data = {"error": str(exc)}
                yield "tool_result", {
                    "name": function_call.name,
                    "status": "failed",
                    "display_value": "Calculation failed",
                }
            call_parts.append(self._types.Part(functionCall=function_call))
            response_parts.append(
                self._types.Part.from_function_response(
                    name=function_call.name,
                    response=response_data,
                )
            )
        follow_up = await asyncio.to_thread(
            self._client.models.generate_content_stream,
            model=self._settings.gemini_model,
            contents=[
                prompt,
                self._types.Content(role="model", parts=call_parts),
                self._types.Content(role="user", parts=response_parts),
            ],
            config=config,
        )
        while True:
            chunk = await asyncio.to_thread(_next_or_none, iter(follow_up))
            if chunk is None:
                break
            usage = provider_usage(chunk)
            if usage is not None:
                self._usage_context().set(usage)
            if getattr(chunk, "text", None):
                yield "token", chunk.text



    async def stream(self, prompt: str) -> AsyncIterator[str]:
        self._usage_context().set(None)
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
            usage = provider_usage(chunk)
            if usage is not None:
                self._usage_context().set(usage)
            if getattr(chunk, "text", None):
                yield chunk.text

