"""Safe wrapper around the five public ptobench commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from .context import RunContext
from .events import EventLedger
from .protocol import ProtocolError
from .source import fingerprint


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class ActionObservation:
    schema_version: int
    method_action_id: str
    operation: str
    profile_mode: str | None
    workspace: str
    source_before: str
    started_at: str
    ended_at: str
    exit_code: int | None
    cli_result: Any
    source_after: str | None
    stdout_path: str
    stderr_path: str
    result_path: str
    parse_status: str
    fork_workspace: str | None = None

    @property
    def bench_event_seq(self) -> Any:
        return self.cli_result.get("event_seq") if isinstance(self.cli_result, Mapping) else None

    @property
    def snapshot_id(self) -> Any:
        return self.cli_result.get("snapshot_id") if isinstance(self.cli_result, Mapping) else None

    @property
    def remaining_hardware_actions(self) -> Any:
        return self.cli_result.get("remaining_hardware_actions") if isinstance(self.cli_result, Mapping) else None

    def to_dict(self) -> dict[str, Any]:
        value = self.__dict__.copy()
        result = self.cli_result if isinstance(self.cli_result, Mapping) else {}
        # Keep Bench fields as observations supplied by the public CLI.  In
        # particular, profile output is allowed to omit correctness/latency.
        value["bench_event_seq"] = result.get("event_seq")
        value["snapshot_id"] = result.get("snapshot_id")
        value["remaining_hardware_actions"] = result.get("remaining_hardware_actions")
        return value


class PublicActionRunner:
    ALLOWED = {"evaluate", "profile", "fork"}
    PROFILE_MODES = {"runtime", "pmu", "compiler"}

    def __init__(self, context: RunContext, ledger: EventLedger | None = None,
                 *, executable: str | None = None):
        self.context = context
        self.ledger = ledger or EventLedger(context)
        self.executable = executable or shutil.which("ptobench") or "ptobench"

    def command(self, operation: str, profile_mode: str | None = None,
                workspace: Path | None = None) -> list[str]:
        if operation not in self.ALLOWED:
            raise ProtocolError("operation is not allowed")
        if operation == "profile":
            if profile_mode not in self.PROFILE_MODES or profile_mode not in self.context.profiling_modes:
                raise ProtocolError("profile mode is not available")
            args = ["profile", profile_mode]
        else:
            if profile_mode is not None:
                raise ProtocolError("profile mode is only valid for profile")
            args = [operation]
        selected = self.context.validate_workspace(workspace)
        argv = [self.executable, *args, str(selected)]
        return argv

    @staticmethod
    def _last_json(stdout: bytes) -> tuple[str, Any]:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            return "invalid", None
        try:
            value = json.loads(lines[-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "invalid", None
        return ("valid", value) if isinstance(value, dict) else ("invalid", None)

    def run_action(self, operation: str, profile_mode: str | None = None,
                   workspace: Path | None = None, *, timeout: float | None = None, cancel=None,
                   action_id: str | None = None) -> ActionObservation:
        selected = self.context.validate_workspace(workspace)
        argv = self.command(operation, profile_mode, selected)
        action_id = action_id or str(uuid.uuid4())
        try:
            uuid.UUID(action_id)
        except (ValueError, TypeError) as exc:
            raise ProtocolError("action_id must be a UUID") from exc
        action_dir = self.context.work_root / "actions" / action_id
        action_dir.mkdir(parents=True, exist_ok=False)
        stdout_file = action_dir / "stdout.log"
        stderr_file = action_dir / "stderr.log"
        result_file = action_dir / "result.json"
        before = fingerprint(selected)
        started = _now()
        self.ledger.append("action_started", workspace=selected, source_hash=before.content_hash,
                           method_action_id=action_id, payload_ref=str(result_file.relative_to(self.context.session_root)))
        exit_code: int | None = None
        out = err = b""
        try:
            if cancel is not None and cancel.is_set():
                raise ProtocolError("Public action cancelled before spawn")
            # Drain directly to durable files; no pipe buffer can lose an interrupted call.
            with stdout_file.open("wb", buffering=0) as out_stream, stderr_file.open("wb", buffering=0) as err_stream:
                environment = {**os.environ, 'PTOBENCH_SESSION_ROOT':str(self.context.session_root),
                    'PTOBENCH_SESSION_ID':self.context.session_id, 'PTOBENCH_CASE_ID':self.context.case_id,
                    'PTOBENCH_WORKSPACE':str(selected)}
                proc = subprocess.Popen(argv, cwd=selected, env=environment, shell=False,
                                        stdin=subprocess.DEVNULL, stdout=out_stream, stderr=err_stream)
                deadline = time.monotonic() + timeout if timeout is not None else None
                stopped_at = None
                while proc.poll() is None:
                    stopping = (cancel is not None and cancel.is_set()) or (deadline is not None and time.monotonic() >= deadline)
                    if stopping and stopped_at is None:
                        proc.terminate()
                        stopped_at = time.monotonic()
                    if stopped_at is not None and time.monotonic() - stopped_at > 20:
                        proc.kill()
                    try:
                        proc.wait(timeout=.05)
                    except subprocess.TimeoutExpired:
                        pass
                exit_code = proc.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            stderr_file.write_bytes(str(exc).encode("utf-8", "replace"))
            exit_code = None
        out = stdout_file.read_bytes() if stdout_file.exists() else b""
        if not stdout_file.exists(): stdout_file.write_bytes(b"")
        if not stderr_file.exists(): stderr_file.write_bytes(b"")
        parse_status, parsed = self._last_json(out)
        ended = _now()
        try:
            after = fingerprint(selected).content_hash
        except ProtocolError:
            after = None
        fork_workspace = None
        if operation == "fork" and isinstance(parsed, Mapping):
            candidate = parsed.get("workspace") or parsed.get("new_workspace")
            if isinstance(candidate, str):
                try:
                    fork_workspace = str(self.context.validate_workspace(Path(candidate)))
                except ProtocolError:
                    fork_workspace = None
        observation = ActionObservation(1, action_id, operation, profile_mode, str(selected), before.content_hash,
                                       started, ended, exit_code, parsed, after,
                                       str(stdout_file.relative_to(self.context.session_root)),
                                       str(stderr_file.relative_to(self.context.session_root)),
                                       str(result_file.relative_to(self.context.session_root)), parse_status,
                                       fork_workspace)
        result_file.write_text(json.dumps(observation.to_dict(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
        event_type = "action_finished" if exit_code == 0 and parse_status == "valid" else "action_failed"
        self.ledger.append(event_type, workspace=selected, source_hash=after or before.content_hash,
                           method_action_id=action_id, payload_ref=str(result_file.relative_to(self.context.session_root)),
                           bench_event_seq=observation.bench_event_seq,
                           payload={"exit_code": exit_code, "parse_status": parse_status})
        return observation
