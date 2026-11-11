"""Prepare and collect the foundation's method-owned files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .context import RunContext
from .events import EventLedger
from .protocol import ProtocolError, response
from .source import fingerprint


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ProtocolError("resource path must not be a symlink")
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _readonly(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode & ~0o222)
    except OSError as exc:
        raise ProtocolError("resource could not be made read-only", "infrastructure_error") from exc


def _resource_manifest(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append({"path": path.relative_to(root).as_posix(), "sha256": "sha256:" + digest})
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"schema_version": 1, "algorithm": "sha256", "files": entries,
            "content_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest()}


def prepare(context: RunContext, request: Mapping[str, Any]) -> dict[str, Any]:
    method = context.method_root
    work = context.work_root
    for path in (work / "actions", work / "logs", work / "resources"):
        path.mkdir(parents=True, exist_ok=True)
    source = fingerprint(context.workspace).to_dict()
    config = {"schema_version": 1, "foundation": "sluice-foundation-v1", "context": context.public_dict(),
              "experiment": request["experiment"], "source": source}
    _write(method / "config.json", config)
    _write(method / "README.md", "SLUICE foundation scaffold. Public workspace and ptobench actions only.\n")
    manifest = _resource_manifest(method / "resources")
    _write(method / "resource_manifest.json", manifest)
    _write(method / "source.json", source)
    for path in (method / "config.json", method / "README.md", method / "resource_manifest.json", method / "source.json"):
        _readonly(path)
    ledger = EventLedger(context)
    ledger.append("run_prepared", source_hash=source["content_sha256"], payload_ref="method/config.json")
    return response(records={"method/config.json": config, "method/resource_manifest.json": manifest,
                             "method/source.json": source, "method/README.md": (method / "README.md").read_text()},
                    archives={"resources": "method/resources"})


def collect(context: RunContext, request: Mapping[str, Any]) -> dict[str, Any]:
    ledger = EventLedger(context)
    events = ledger.events()
    missing: list[str] = []
    actions = context.work_root / "actions"
    started_ids = {item.get("method_action_id") for item in events
                   if item.get("event_type") == "action_started" and item.get("method_action_id")}
    finished_ids = {item.get("method_action_id") for item in events
                    if item.get("event_type") in {"action_finished", "action_failed"} and item.get("method_action_id")}
    if actions.exists():
        for directory in sorted(p for p in actions.iterdir() if p.is_dir()):
            result = directory / "result.json"
            if not result.exists() or (directory.name in started_ids and directory.name not in finished_ids):
                missing.append(directory.name)
                if not any(item.get("event_type") == "interrupted" and item.get("method_action_id") == directory.name
                           for item in events):
                    ledger.append("interrupted", workspace=context.workspace, method_action_id=directory.name,
                                  payload_ref=str(directory.relative_to(context.session_root)))
    status = "missing" if missing else "recorded"
    event_hash = ledger.finalize()
    summary = {"schema_version": 1, "recording": {"status": status}, "event_count": len(ledger.events()),
               "action_count": len(list(actions.iterdir())) if actions.exists() else 0,
               "missing": missing, "close_reason": request.get("close_reason"), "events_sha256": event_hash}
    _write(context.method_root / "summary.json", summary)
    _readonly(context.method_root / "summary.json")
    return response(records={"method/summary.json": summary}, archives={"method_work": "method/work"},
                    recording={"status": status, "missing": missing})
