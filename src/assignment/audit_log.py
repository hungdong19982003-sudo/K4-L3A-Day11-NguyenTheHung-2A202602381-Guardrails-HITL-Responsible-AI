"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id/user_id."""
        import time
        req_key = request_id or f"{user_id}_{len(self.logs)}_{time.time()}"
        self._open[req_key] = {
            "start_time": time.time(),
            "timestamp": utc_now_iso(),
            "user_id": user_id,
            "input": text,
            "request_id": req_key,
        }
        return req_key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Store output, layer decision, latency; append to self.logs."""
        import time
        open_entry = self._open.pop(request_id, None) if request_id else None
        latency_ms = 0.0
        start_iso = utc_now_iso()
        input_text = ""
        if open_entry:
            latency_ms = (time.time() - open_entry["start_time"]) * 1000.0
            start_iso = open_entry.get("timestamp", start_iso)
            input_text = open_entry.get("input", "")

        log_entry = {
            "request_id": request_id,
            "timestamp": start_iso,
            "user_id": user_id,
            "input": input_text,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round(latency_ms, 2),
        }
        self.logs.append(log_entry)
        return log_entry

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path_str = filepath or default_audit_log_path()
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
