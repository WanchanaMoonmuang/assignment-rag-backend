from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from typing import Any, AsyncIterator
from uuid import uuid4

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.calculator import CalculatorError, calculate

from app.settings import Settings

FALLBACK_ANSWER = "I could not generate an answer. Please try again."
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    from google.genai import errors as genai_errors

    return isinstance(exc, genai_errors.APIError) and exc.code in _RETRYABLE_STATUS_CODES
SYSTEM_INSTRUCTION = (
    "Answer in the language of the user's latest question. "
    "When document context is provided, support document-based claims with inline markers "
    "such as [1] that match the numbered sources. "
    "You may use general knowledge when document context is incomplete, but put those claims "
    "under a 'General knowledge' heading without source markers. "
    "Do not claim that uncited general knowledge came from a document. Keep the answer clear and concise. "
    "Only use the calculator tool for questions that require an actual numeric computation; "
    "answer conceptual or document questions directly without it."
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
        self._embed_rate_lock = asyncio.Lock()
        self._last_embed_call_at: float | None = None

    def _usage_context(self) -> ContextVar[dict[str, int] | None]:
        context = getattr(self, "_last_usage", None)
        if context is None:
            context = ContextVar("last_usage", default=None)
            self._last_usage = context
        return context


    def last_usage(self) -> dict[str, int] | None:
        return self._usage_context().get()

    async def _throttle_embed_rate(self) -> None:
        # Vertex AI's embed_content quota is per-minute, not a transient burst
        # limit — pacing calls up front avoids exhausting it, since retrying
        # after a 429 can't outlast a quota window that's genuinely used up.
        min_interval = 60.0 / self._settings.gemini_embed_requests_per_minute
        async with self._embed_rate_lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            if self._last_embed_call_at is not None:
                wait = min_interval - (now - self._last_embed_call_at)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_embed_call_at = loop.time()

    async def _embed_one_with_retry(self, content: Any, task_type: str) -> Any:
        # Vertex AI's embedContent API rejects more than one content per call for
        # this model ("only supports one content at a time"), so calls stay 1:1
        # with texts; only transient errors are retried, one call at a time.
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable_gemini_error),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            reraise=True,
        ):
            with attempt:
                return await asyncio.to_thread(
                    self._client.models.embed_content,
                    model=self._settings.gemini_embedding_model,
                    contents=[content],
                    config=self._types.EmbedContentConfig(
                        output_dimensionality=self._settings.gemini_embedding_dimensions,
                        task_type=task_type,
                    ),
                )

    async def embed_texts(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for text in texts:
            content = self._types.Content(parts=[self._types.Part.from_text(text=text)])
            await self._throttle_embed_rate()
            result = await self._embed_one_with_retry(content, task_type)
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


    def _calculator_config(self) -> Any:
        declaration = self._types.FunctionDeclaration(
            name="calculator",
            description=(
                "Evaluate an explicit arithmetic expression (e.g. '12*7.5', '(3+4)/2'). "
                "Only call this when the user asks for a concrete numeric calculation. "
                "Do not call it for conceptual, definitional, or document questions."
            ),
            parametersJsonSchema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
        return self._types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self._settings.gemini_temperature,
            tools=[self._types.Tool(functionDeclarations=[declaration])],
        )

    def _calculator_response_part(self, function_call: Any) -> tuple[Any, dict[str, Any]]:
        arguments = dict(function_call.args or {})
        try:
            result = calculate(str(arguments.get("expression", "")))
            record = {"name": function_call.name, "arguments": arguments, "result": result}
            response_data: dict[str, Any] = {"result": result}
        except CalculatorError as exc:
            record = {"name": function_call.name, "arguments": arguments, "error": str(exc)}
            response_data = {"error": str(exc)}
        part = self._types.Part.from_function_response(name=function_call.name, response=response_data)
        return part, record

    async def generate_with_calculator(
        self, prompt: str
    ) -> tuple[str, list[dict[str, Any]], dict[str, int] | None]:
        config = self._calculator_config()
        contents: list[Any] = [
            self._types.Content(role="user", parts=[self._types.Part.from_text(text=prompt)])
        ]
        activity: list[dict[str, Any]] = []
        response: Any = None
        # A multi-step question (e.g. "average X per year, then compare") can need more
        # than one calculator call in sequence -- the model's follow-up turn may itself
        # request another call instead of returning final text. Loop (bounded) instead
        # of assuming one round, or a chained follow-up silently returns empty text.
        for _ in range(self._settings.gemini_max_tool_rounds):
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._settings.gemini_model,
                contents=contents,
                config=config,
            )
            function_calls = list(getattr(response, "function_calls", None) or [])
            if not function_calls:
                return (getattr(response, "text", "") or "").strip(), activity, provider_usage(response)

            response_parts = []
            for function_call in function_calls:
                part, record = self._calculator_response_part(function_call)
                response_parts.append(part)
                activity.append(record)

            # Reuse the model's own returned content (not a hand-built Part) so the
            # required thought_signature on the function-call part is preserved.
            contents.append(response.candidates[0].content)
            contents.append(self._types.Content(role="user", parts=response_parts))

        return (getattr(response, "text", "") or "").strip(), activity, provider_usage(response)


    async def stream_with_calculator(self, prompt: str) -> AsyncIterator[tuple[str, Any]]:
        config = self._calculator_config()
        self._usage_context().set(None)
        contents: list[Any] = [
            self._types.Content(role="user", parts=[self._types.Part.from_text(text=prompt)])
        ]
        # See generate_with_calculator: a chained follow-up call must also be handled,
        # bounded, instead of assuming the model is done after one round of tool use.
        for _ in range(self._settings.gemini_max_tool_rounds):
            stream = await asyncio.to_thread(
                self._client.models.generate_content_stream,
                model=self._settings.gemini_model,
                contents=contents,
                config=config,
            )
            function_calls: list[Any] = []
            call_parts: list[Any] = []
            while True:
                chunk = await asyncio.to_thread(_next_or_none, iter(stream))
                if chunk is None:
                    break
                usage = provider_usage(chunk)
                if usage is not None:
                    self._usage_context().set(usage)
                if getattr(chunk, "text", None):
                    yield "token", chunk.text
                chunk_calls = getattr(chunk, "function_calls", None) or []
                if chunk_calls:
                    function_calls.extend(chunk_calls)
                    candidates = getattr(chunk, "candidates", None) or []
                    if candidates and candidates[0].content and candidates[0].content.parts:
                        # Preserve the model's own parts (incl. thought_signature) rather
                        # than reconstructing Part(functionCall=...) from scratch.
                        call_parts.extend(candidates[0].content.parts)
            if not function_calls:
                return
            response_parts = []
            for function_call in function_calls:
                yield "tool_call", {"name": function_call.name, "status": "requested"}
                part, record = self._calculator_response_part(function_call)
                response_parts.append(part)
                yield "tool_result", (
                    {
                        "name": function_call.name,
                        "arguments": record["arguments"],
                        "status": "completed",
                        "display_value": str(record["result"]),
                    }
                    if "result" in record
                    else {
                        "name": function_call.name,
                        "arguments": record["arguments"],
                        "status": "failed",
                        "display_value": "Calculation failed",
                    }
                )
            contents.append(self._types.Content(role="model", parts=call_parts))
            contents.append(self._types.Content(role="user", parts=response_parts))



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

