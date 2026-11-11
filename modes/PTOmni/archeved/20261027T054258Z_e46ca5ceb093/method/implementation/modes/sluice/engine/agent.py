"""One native process; archive raw streams and sessions without detached workers."""

from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
from . import backend
from .controller import save
from ..foundation.context import in_current_process_group
from ..foundation.events import EventLedger
from ..foundation.protocol import ProtocolError
from ..foundation.source import fingerprint


def descendants(pid):
    """Capture PID/birth pairs for only the native process and its current descendants."""
    rows = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            parts = path.read_text().rsplit(")", 1)[1].split()
            rows[int(path.parent.name)] = (int(parts[1]), parts[19])
        except (OSError, ValueError, IndexError):
            pass
    found = {pid}
    changed = True
    while changed:
        old = len(found)
        found.update(p for p, (parent, _) in rows.items() if parent in found)
        changed = len(found) != old
    return [(p, rows[p][1]) for p in found if p in rows]


def stop(owned, sig):
    for pid, birth in owned:
        try:
            now = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            if now == birth:
                os.kill(pid, sig)
        except (OSError, IndexError):
            pass


class NativeAgent:
    def run(self, context, config):
        if not in_current_process_group():
            raise ProtocolError("Method process group does not match Launcher")
        directory = context.method_root / "agent/main"
        directory.mkdir(parents=True, exist_ok=True)
        prompt = (context.method_root / "prompt.txt").read_text()
        argv, additions = backend.command(config, prompt)
        environment = {**os.environ, **additions}
        if config["profile"]["backend"] != "fake":
            environment.update(backend.provider.credentials(config["profile"]))
        if config['profile']['backend'] == 'codex':
            # Keep native history in this run's mounted directory. Bring only
            # selected authentication/settings, never host conversation files.
            source = backend.recording.config_directory(config['profile'])
            store = Path(environment['CODEX_HOME'])
            store.mkdir(parents=True, exist_ok=True)
            for name in ('auth.json', 'config.toml'):
                if (source / name).is_file() and not (store / name).exists():
                    shutil.copyfile(source / name, store / name)
                    (store / name).chmod(0o600)
        save(
            directory / "command.json",
            {
                "argv": argv,
                "profile_sha256": config["profile"].get("sha256"),
                "cwd": str(context.workspace),
                "process_group": os.getpgrp(),
                "source_before": fingerprint(context.workspace).content_hash,
            },
        )
        (directory / "prompt.txt").write_text(prompt)
        ledger = EventLedger(context)
        ledger.append(
            "agent_started",
            source_hash=fingerprint(context.workspace).content_hash,
            payload_ref="method/agent/main/command.json",
        )
        cancel = threading.Event()
        previous = {}

        def requested(*_):
            save(context.work_root / "closed.json", {"reason": "Method received a stop signal"})
            cancel.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, requested)
        boundary = None
        process = None
        owned = []
        stopped = None
        try:
            if config['profile']['backend'] != 'fake':
                from .agent_boundary import AgentBoundary
                boundary = AgentBoundary(context, config)
                argv = boundary.command(argv, environment)
            with (
                (directory / "stdout.log").open("wb", buffering=0) as out,
                (directory / "stderr.log").open("wb", buffering=0) as err,
            ):
                process = subprocess.Popen(
                    argv,
                    cwd=context.workspace,
                    env=environment,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                )
                save(
                    directory / "process.json", {"pid": process.pid, "process_group": os.getpgid(process.pid)}
                )
                while process.poll() is None:
                    if cancel.is_set() and stopped is None:
                        owned = descendants(process.pid)
                        stop(owned, signal.SIGTERM)
                        stopped = time.monotonic()
                    if stopped is not None and time.monotonic() - stopped > 22:
                        stop(owned, signal.SIGKILL)
                    try:
                        process.wait(timeout=0.05)
                    except subprocess.TimeoutExpired:
                        pass
                code = process.wait()
        except Exception as error:
            save(directory / 'infrastructure_error.json', {'type': type(error).__name__, 'message': str(error)})
            raise
        finally:
            closed = context.work_root / "closed.json"
            previous_close = json.loads(closed.read_text()) if closed.exists() else {}
            save(closed, {**previous_close, "execution_closed": True, "agent_stopped": True})
            if process is not None and process.poll() is None:
                stop(descendants(process.pid), signal.SIGKILL)
                process.wait()
            if owned:
                stop(owned, signal.SIGKILL)
            if boundary is not None:
                boundary.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        result = {
            "schema_version": 1,
            "exit_code": code,
            "cancelled": cancel.is_set(),
            "source_after": fingerprint(context.workspace).content_hash,
        }
        if config["profile"]["backend"] == "fake":
            native = {
                "status": "recorded",
                "test_only": True,
                "native_session_ids": [],
                "events_path": "stdout.log",
                "backend": "fake",
            }
        else:
            _, rendered = backend.render(config["profile"], context.workspace, directory, prompt)
            native_profile = {**config["profile"], "env": rendered}
            native = backend.recording.collect(
                directory,
                native_profile,
                context.workspace,
                started=True,
                close_reason="interrupted" if cancel.is_set() else "agent_exit",
            )
        answer = None
        native_error = None
        for event in backend.recording._json_lines(directory / "stdout.log"):
            if event.get("type") == "message_end":
                message = event.get("message", {})
                if message.get("role") == "assistant":
                    answer = message.get("content")
                    if message.get("stopReason") in ("error", "aborted"):
                        native_error = message.get("errorMessage") or message["stopReason"]
            if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message":
                answer = event["item"].get("text")
            if event.get("type") in ("turn.failed", "error"):
                native_error = event.get("error", event.get("message", "Native error"))
        result.update(
            answer=answer,
            native_error=native_error,
            is_error=code != 0 or cancel.is_set() or native_error is not None,
        )
        save(directory / "recording.json", native)
        save(directory / "result.json", result)
        ledger.append(
            "agent_finished", source_hash=result["source_after"], payload_ref="method/agent/main/result.json"
        )
        return result
