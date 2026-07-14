from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("app.metrics")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def emit(event: str, **fields: Any) -> None:
    try:
        logger.info(json.dumps({"event": event, **fields}, default=str, separators=(",", ":")))
    except Exception:
        pass
