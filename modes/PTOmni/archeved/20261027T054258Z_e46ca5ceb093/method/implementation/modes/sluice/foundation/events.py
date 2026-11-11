"""Append-only method event ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from .context import RunContext
from .protocol import ProtocolError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventLedger:
    def __init__(self, context: RunContext, path: str | Path | None = None):
        self.context = context
        self.path = Path(path) if path is not None else context.work_root / "events.jsonl"
        if self.path.is_symlink():
            raise ProtocolError("event ledger path must not be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._event_ids: set[str] = set()
        if self.path.exists():
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict) and isinstance(item.get("event_id"), str):
                            self._event_ids.add(item["event_id"])
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ProtocolError("event ledger is unreadable", "infrastructure_error") from exc

    def append(self, event_type: str, *, workspace: str | Path | None = None,
               source_hash: str | None = None, method_action_id: str | None = None,
               bench_event_seq: int | None = None, payload_ref: str | None = None,
               payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        allowed = {"run_started", "run_prepared", "run_finished", "action_started",
                   "action_finished", "action_failed", "interrupted", "incomplete",
                   "tool_started", "tool_finished", "hypothesis_registered", "attempt_applied",
                   "decision", "step_accepted", "agent_started", "agent_finished"}
        if event_type not in allowed:
            raise ProtocolError("unsupported event type")
        if payload_ref is not None:
            reference = Path(payload_ref)
            if reference.is_absolute() or ".." in reference.parts:
                raise ProtocolError("payload_ref must be a run-relative path")
        event_id = str(uuid.uuid4())
        while event_id in self._event_ids:
            event_id = str(uuid.uuid4())
        event = {"schema_version": 1, "event_id": event_id, "event_type": event_type,
                 "time": utc_now(), "case_id": self.context.case_id,
                 "workspace": str(workspace or self.context.workspace),
                 "source_hash": source_hash or "", "method_action_id": method_action_id,
                 "bench_event_seq": bench_event_seq, "payload_ref": payload_ref}
        if payload:
            forbidden = {"best", "opportunity", "stop", "causal", "winner"}
            if any(key.lower() in forbidden for key in payload):
                raise ProtocolError("event payload contains policy state")
            event["payload"] = dict(payload)
        serialized = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        try:
            if self.path.is_symlink():
                raise OSError("event ledger path became a symlink")
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
                stream.flush()
                import os
                os.fsync(stream.fileno())
        except OSError as exc:
            raise ProtocolError("event ledger write failed", "infrastructure_error") from exc
        self._event_ids.add(event_id)
        return event

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("event ledger is unreadable", "infrastructure_error") from exc

    def sha256(self) -> str:
        try:
            digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ProtocolError("event ledger could not be hashed", "infrastructure_error") from exc
        return "sha256:" + digest

    def finalize(self) -> str:
        value = self.sha256()
        target = self.path.with_name("events_sha256")
        try:
            target.write_text(value + "\n", encoding="ascii")
        except OSError as exc:
            raise ProtocolError("event ledger hash could not be saved", "infrastructure_error") from exc
        return value
