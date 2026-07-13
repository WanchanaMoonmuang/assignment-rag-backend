FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app

ENV PATH="/app/.venv/bin:$PATH" \
    PORT=8080

CMD ["sh", "-c", "python -m app.worker & worker=$!; uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} & api=$!; stop() { kill -TERM \"$worker\" \"$api\" 2>/dev/null || true; wait \"$worker\" 2>/dev/null || true; wait \"$api\" 2>/dev/null || true; exit \"$1\"; }; trap 'stop 0' INT TERM; while kill -0 \"$worker\" 2>/dev/null && kill -0 \"$api\" 2>/dev/null; do sleep 1; done; stop 1"]
