from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger("app.metrics")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    # Cloud Run captures stdout, not stderr (logging.StreamHandler's default) — and
    # force line-buffering so events flush immediately in a long-running, non-interactive
    # process instead of sitting in a block buffer until it fills or the process exits.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def emit(event: str, **fields: Any) -> None:
    try:
        logger.info(json.dumps({"event": event, **fields}, default=str, separators=(",", ":")))
    except Exception:
        pass
