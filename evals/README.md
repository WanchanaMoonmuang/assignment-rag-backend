# RAG evaluation harness (RAGAS)

Scores the real `/api/chat` retrieval + generation path against a golden set, using
[RAGAS](https://docs.ragas.io/) with a Gemini judge on Vertex AI.

## Setup

```bash
uv sync --extra eval
```

`eval_judge_model` in `app/settings.py` (or `EVAL_JUDGE_MODEL` in `.env`) must name a model
actually enabled on your Vertex AI project/location — check with:

```bash
uv run python -c "
from google import genai
c = genai.Client(vertexai=True, project='<project>', location='<location>')
for m in c.models.list():
    print(m.name)
"
```

The judge should be a stronger model than the one you serve (`gemini_model` in `.env`), so it
isn't grading its own homework.

## Filling the golden set

Edit `evals/golden_set.json` — one array per file in `samples/`, ~5 rows each:

```json
{"question": "...", "ground_truth": "..."}
```

`ground_truth` must be a **complete reference answer**, not "see page 3" — RAGAS decomposes it
into individual claims to score `context_recall` and `faithfulness`. Rows with an empty
`question` or `ground_truth` are skipped (that's how the unfilled template rows are ignored).

## Running

Two modes:

- **`--mode ingest`** (default) — ingests every `samples/` file referenced in the golden set
  through the real extraction → embedding → storage path, waits for the Atlas search/vector
  indexes to catch up, scores, then deletes the documents/chunks it created. Refuses to run if
  the chunk collection already has data (pass `--force` to override) — leftover docs skew
  `context_precision`/`context_recall`.
- **`--mode reuse`** — the corpus is already ingested (e.g. via the running API); skips
  ingestion and cleanup entirely and scores against whatever's already in the database. Faster,
  and doesn't touch the embedding quota, but the golden-set questions must correspond to what's
  actually loaded, and precision/recall reflect the *whole* collection, not just `samples/`.

```bash
uv run --extra eval python -m evals.run_eval --mode ingest
uv run --extra eval python -m evals.run_eval --mode reuse
```

Useful flags:
- `--no-answer-relevancy` — drop the one metric that needs embeddings, so nothing competes with
  the ingestion embedding quota.
- `--keep` — don't clean up eval documents after an `ingest` run (for inspecting them by hand).
- `--force` — ingest even if the chunk collection isn't empty.

## Quota note

Ingestion and `answer_relevancy` both use the `gemini-embedding-2` quota
(`gemini_embed_requests_per_minute` in `.env`, commonly much lower than the published ceiling).
They run sequentially — ingestion fully completes before RAGAS starts — so they don't overlap.
A full run can take several minutes; a 429 from a shared/busy project is possible even so.

## Output

Each run writes `evals/results/<timestamp>.json` (per-question scores) and
`evals/results/<timestamp>.md` (aggregate + a per-question table). That directory is
git-ignored.
