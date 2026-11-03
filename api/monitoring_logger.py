import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _log_path() -> Path:
    override = os.getenv("INFERENCE_LOG_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "logs" / "inference" / "inference_log.jsonl"


def log_inference(input_data: dict[str, Any], predicted_rentals: float | None = None, latency_ms: float | None = None,
                  status: str = "success", error: str | None = None, model_name: str | None = None,
                  model_version: str | None = None, run_id: str | None = None, request_id: str | None = None) -> str:
    request_id = request_id or uuid.uuid4().hex[:12]
    event = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "predicted_rentals": predicted_rentals,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    return request_id
