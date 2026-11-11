"""A no-hardware lifecycle harness for validating the foundation."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .context import RunContext, in_current_process_group
from .events import EventLedger
from .protocol import ProtocolError
from .source import fingerprint


class FoundationController:
    def run(self, context: RunContext, ledger: EventLedger) -> Mapping[str, Any]:
        return {"status": "smoke", "source_hash": fingerprint(context.workspace).content_hash}


def run(context: RunContext, request: Mapping[str, Any], controller: FoundationController | None = None) -> dict[str, Any]:
    if not in_current_process_group():
        raise ProtocolError("method process is outside the Launcher process group", "infrastructure_error")
    config_path = context.method_root / "config.json"
    if not config_path.is_file():
        raise ProtocolError("prepared method configuration is missing", "infrastructure_error")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("prepared method configuration is unreadable", "infrastructure_error") from exc
    if (not isinstance(config, Mapping) or config.get("schema_version") != 1
            or config.get("context") != context.public_dict()
            or config.get("experiment") != request.get("experiment")):
        raise ProtocolError("prepared method configuration does not match context", "infrastructure_error")
    ledger = EventLedger(context)
    source = fingerprint(context.workspace)
    ledger.append("run_started", source_hash=source.content_hash, payload_ref="method/config.json")
    try:
        result = dict((controller or FoundationController()).run(context, ledger))
        ledger.append("run_finished", source_hash=source.content_hash, payload_ref="method/config.json",
                      payload={"status": "smoke"})
        return {"schema_version": 1, "status": "completed", "controller": result}
    except BaseException:
        ledger.append("run_finished", source_hash=source.content_hash, payload_ref="method/config.json",
                      payload={"status": "interrupted"})
        raise
