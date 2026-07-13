from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("app.metrics")


def emit(event: str, **fields: Any) -> None:
    try:
        logger.info(json.dumps({"event": event, **fields}, default=str, separators=(",", ":")))
    except Exception:
        pass
