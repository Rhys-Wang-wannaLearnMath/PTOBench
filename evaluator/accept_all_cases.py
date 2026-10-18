"""S7 maintainer acceptance: inventory every Case, run four independent actions.

This is not an Agent command. Never generate fixtures, prepare a Ref, edit a
canonical Case, infer unsupported from generic errors, or combine batch versions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import acceptance_artifacts
import common
import file_tree
import integrity
import profile_modes
import session
import session_actions as actions
import snapshots
import workspace

ITEMS = (("clean", None, "clean"), ("runtime", "runtime", "runtime"),
         ("pmu", "pmu", "hardware"), ("compiler", "compiler", "compile"))
STATUSES = ("PASS", "FAIL", "BLOCKED", "UNSUPPORTED", "TIMEOUT")
FINALIZE_CODE = """import json,sys
sys.path.insert(0,sys.argv[1])
from snapshots import finalize_session
result=finalize_session(sys.argv[2], 'full_v02_acceptance')
print(json.dumps(result.to_dict(),allow_nan=False),flush=True)
raise SystemExit(0 if result.status=='completed' else 1)
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    file_tree.atomic_bytes(path, data)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hash_roots(roots):
    """Full bytes, including .pt files. No links, regeneration or mtime proxies."""
    records, errors = {}, []
    for name, root in sorted(roots.items()):
        try:
            entries = file_tree.scan_tree(root, reject_special=True)
            records[name] = {relative: {k: value[k] for k in ("kind", "sha256", "size") if k in value}
                             for relative, value in entries.items()}
        except (OSError, ValueError, RuntimeError) as error:
            errors.append({"root": name, "error": str(error)})
    return {"roots": records, "errors": errors, "content_sha256": digest(records)}


def differences(before, after):
    a = {f"{root}/{p}": value for root, tree in before["roots"].items() for p, value in tree.items()}
    b = {f"{root}/{p}": value for root, tree in after["roots"].items() for p, value in tree.items()}
    return [{"path": p, "before": a.get(p), "after": b.get(p)}
            for p in sorted(a.keys() | b.keys()) if a.get(p) != b.get(p)]


def last_json(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 262144))
            lines = stream.read().splitlines()
        return json.loads(next(line for line in reversed(lines) if line.strip()))
    except (OSError, ValueError, StopIteration):
        return None


def device_preflight(environment, evidence_dir):
    """Check every requested device; do not silently reduce multi-device Cases."""
    if environment.get("platform") == "cpu":
        return []
    errors = []
    python = common.profile_path(environment.get("python", ""))
    if not python.is_file() or not os.access(python, os.X_OK):
        errors.append("Case Python unavailable")
    for script in environment.get("setup_scripts", []):
        if not common.profile_path(script).is_file():
            errors.append(f"setup script unavailable: {script}")
    ptoas = environment.get("env", {}).get("PTOAS_ROOT")
    if ptoas and not (common.profile_path(ptoas) / "bin/ptoas").is_file():
        errors.append(f"PTOAS unavailable: {ptoas}")
    devices = required_devices(environment)
    if not isinstance(devices, list) or not devices or any(type(d) is not int or d < 0 for d in devices):
        return errors + ["invalid device requirements"]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for device in devices:
        try:
            result = subprocess.run(["npu-smi", "info", "-t", "health", "-i", str(device)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, shell=False)
            (evidence_dir / f"device_{device}.stdout.log").write_bytes(result.stdout)
            (evidence_dir / f"device_{device}.stderr.log").write_bytes(result.stderr)
            if result.returncode or b"OK" not in result.stdout:
                errors.append(f"device {device} unavailable/unhealthy (exit {result.returncode})")
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"device {device} query failed: {error}")
    return errors


def required_devices(environment):
    return environment.get("runtime", {}).get("device_ids",
        environment.get("device_ids", [environment.get("device", 0)]))


def enumerate_cases(bench):
    rows = []
    for path in sorted((bench / "cases").glob("*/*/case.json")):
        relative = path.relative_to(bench).as_posix()
        row = {"case_path": relative, "case_id": None, "certification": "unknown", "environment": {},
               "identity_errors": [], "case_json_sha256": None}
        try:
            value = session._read(path)
            json.dumps(value, allow_nan=False)
            row["case_json_sha256"] = common.sha256_file(path)
            case_id = value.get("case_id")
            row["case_id"] = case_id if isinstance(case_id, str) else None
            valid_id = (isinstance(case_id, str)
                        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*\.[A-Za-z0-9][A-Za-z0-9_-]*", case_id))
            expected = common.case_relative_path(case_id) if valid_id else None
            actual = path.parent.relative_to(bench / "cases")
            if (not valid_id
                    or actual != expected):
                row["identity_errors"].append("case_id_path_mismatch_or_invalid")
            sealed = common.sealed_root() / actual
            if (sealed / "certification.json").is_file():
                cert = session._read(sealed / "certification.json")
                json.dumps(cert, allow_nan=False)
                row["certification"] = cert.get("status", "unknown")
            env_name = value.get("environment")
            if isinstance(env_name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", env_name):
                env_path = bench / "environments" / f"{env_name}.json"
                if env_path.is_file():
                    row["environment"] = session._read(env_path)
                    json.dumps(row["environment"], allow_nan=False)
                    if not isinstance(row["environment"], dict) or not isinstance(row["environment"].get("runtime", {}), dict):
                        raise ValueError("invalid environment object")
            seed_relative = value.get("workspace")
            if isinstance(seed_relative, str):
                seed = session._inside(path.parent, seed_relative)
                if seed.is_dir():
                    row["source_hash"] = workspace.source_manifest(seed)["content_sha256"]
        except (OSError, ValueError, TypeError, AttributeError, integrity.InfrastructureError) as error:
            # Invalid metadata must still produce four finite JSON records.
            row["environment"] = {}
            row["certification"] = "unknown"
            row["identity_errors"].append(f"inventory_parse_error:{error}")
        rows.append(row)
    counts = Counter(row["case_id"] for row in rows if isinstance(row["case_id"], str))
    for row in rows:
        if isinstance(row["case_id"], str) and counts[row["case_id"]] > 1:
            row["identity_errors"].append("duplicate_case_id")
    return sorted(rows, key=lambda row: (str(row["case_id"] or ""), row["case_path"]))


def unsupported_evidence(raw, mode, stderr):
    """Only explicit profiling capability failures, never generic unsupported text."""
    error = raw.get("error") if isinstance(raw, dict) else None
    if (isinstance(error, dict) and error.get("code") == "profiling_unsupported"
            and error.get("mode") in {mode, session.PUBLIC_MODES.get(mode)}):
        return {"source": "evaluator_json", "error": error}
    names = {mode, session.PUBLIC_MODES.get(mode)}
    for line in stderr.splitlines():
        if re.match(r"^(?:ProfileNotSupportedError|ProfilingNotSupportedError):", line.strip()):
            if any(name and re.search(r"\b" + re.escape(name) + r"\b", line) for name in names):
                return {"source": "typed_exception", "exception": line.strip()}
    return None


class AcceptanceRunner:
    def __init__(self, output, *, timeout=1800, device_probe=device_preflight):
        self.bench = common.BENCHMARK_ROOT
        self.output = file_tree.absolute_path(output)
        self.timeout = timeout
        self.device_probe = device_probe
        self.rows = []
        self.finals = []
        self.stop_reason = None
        self.cases = []
        self.device_cache = {}
        self.frozen_assets = None
        self.code_roots = {name: self.bench / name for name in ("evaluator", "launcher", "ptobench", "pto-launcher")
                           if (self.bench / name).exists()}
        self.code_roots["golden"] = common.REPO_ROOT / "golden"
        self.asset_roots = {"cases": self.bench / "cases", "sealed": common.sealed_root(),
                            "environments": self.bench / "environments"}

    def append(self, row):
        if row["status"] == "PASS":
            row["issue_owner"] = None
        elif row["status"] == "BLOCKED" and str(row["reason"]).startswith(("case_preflight", "case_id", "duplicate_case_id", "inventory_parse")):
            row["issue_owner"] = "case_assets_or_migration"
        elif row["status"] == "BLOCKED" and str(row["reason"]).startswith(("device_or_toolchain", "mode_disabled")):
            row["issue_owner"] = "environment_or_profile_configuration"
        else:
            row["issue_owner"] = "public_execution_or_case_runtime_requires_diagnosis"
        json.dumps(row, allow_nan=False)
        with (self.output / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        print(f"[{len(self.rows)}/{len(self.cases)*4}] {row['case_id'] or row['case_path']} {row['item']}: "
              f"{row['status']} {row['reason'] or ''}", flush=True)

    def base_row(self, case, item, public, internal):
        return {"schema_version": 1, "case_id": case["case_id"], "case_path": case["case_path"],
            "certification": case["certification"], "item": item, "public_mode": public, "internal_mode": internal,
            "status": "BLOCKED", "supported": None, "correctness": "not_run", "integrity_passed": None,
            "artifact_check": None, "reason": None, "error_code": None, "argv": None, "exit_code": None,
            "elapsed_seconds": 0, "environment": case["environment"],
            "devices": required_devices(case["environment"]),
            "source_hash": case.get("source_hash"), "session_id": None, "session_root": None, "event_seq": None,
            "stdout_path": None, "stderr_path": None, "result_path": None, "raw_result_path": None,
            "completed_at": now()}

    def blocked_case(self, case, reason):
        for item, public, internal in ITEMS:
            row = self.base_row(case, item, public, internal)
            row["reason"] = reason
            self.append(row)

    def run_process(self, argv, cwd, environment, directory, root=None):
        directory.mkdir(parents=True, exist_ok=False)
        save(directory / "command.json", {"argv": argv, "cwd": str(cwd), "timeout_seconds": self.timeout})
        start = time.monotonic()
        timed_out, cleanup_error = False, None
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True, shell=False)
            identity = session.process_identity(process.pid)
            try:
                process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        actions._stop_group(identity, process)
                    except Exception as error:
                        cleanup_error = str(error)
                        self.stop_reason = f"unconfirmed_process_cleanup:{error}"
            except BaseException:
                actions._stop_group(identity, process)
                raise
            try:
                if identity is not None and session._group_alive(identity):
                    actions._stop_group(identity, process)
                if root is not None:
                    record = session.read_session(root)
                    if record["active_action"] is not None:
                        if not timed_out:
                            raise integrity.InfrastructureError("CLI exited with unfinished hardware action")
                        # SIGINT normally performs this inside the CLI. Recover a
                        # killed CLI's independently grouped Evaluator explicitly.
                        with session.execution_lock():
                            actions.recover_global_action()
                            session.reconcile_active_action(root)
                    if session.read_session(root)["active_action"] is not None:
                        raise integrity.InfrastructureError("cannot confirm hardware stopped")
            except Exception as error:
                cleanup_error = str(error)
                self.stop_reason = f"unconfirmed_process_cleanup:{error}"
        value = last_json(directory / "stdout.log")
        result = {"argv": argv, "exit_code": process.returncode, "elapsed_seconds": time.monotonic()-start,
            "timed_out": timed_out, "cleanup_error": cleanup_error, "value": value,
            "process_identity": identity, "cleanup_confirmed": cleanup_error is None,
            "stdout_path": str(directory / "stdout.log"), "stderr_path": str(directory / "stderr.log")}
        save(directory / "process_result.json", result)
        return result

    def cli(self, handle, args, target, directory):
        environment = os.environ.copy()
        environment.update(PTOBENCH_SESSION_ROOT=str(handle.session_root), PYTHONDONTWRITEBYTECODE="1")
        return self.run_process([sys.executable, str(self.bench / "ptobench"), *args, str(target)],
            target, environment, directory, handle.session_root)

    def inspect_action(self, case, handle, item, public, internal, process):
        row = self.base_row(case, item, public, internal)
        row.update({k: process[k] for k in ("argv", "exit_code", "elapsed_seconds", "stdout_path", "stderr_path")})
        row.update(session_id=handle.session_root.name, session_root=str(handle.session_root), correctness="unknown", status="FAIL")
        value = process["value"]
        row["result_path"] = str(Path(process["stdout_path"]).parent / "process_result.json")
        if process["timed_out"]:
            row.update(status="TIMEOUT", reason="command_timeout", error_code="timeout")
        elif process["cleanup_error"]:
            row.update(reason=process["cleanup_error"], error_code="infrastructure_error")
        if not isinstance(value, dict) or type(value.get("event_seq")) is not int:
            row["reason"] = row["reason"] or "missing_or_invalid_session_json"
            return row
        row.update(event_seq=value["event_seq"], error_code=value.get("error_code"))
        try:
            record = session.read_session(handle.session_root)
            events = [json.loads(line) for line in (handle.session_root / "events.jsonl").read_text().splitlines()]
            matches = [event for event in events if event["seq"] == value["event_seq"]]
            if len(matches) != 1:
                raise ValueError("missing or duplicate authority event")
            event = matches[0]
            if event["action"] != ("evaluate" if public is None else "profile") or event["mode"] != public:
                raise ValueError("event action/mode mismatch")
            action_dir = handle.session_root / "staging" / f"action_{event['seq']:06d}"
            raw_path = action_dir / "result.json"
            raw = session._read(raw_path) if raw_path.is_file() else event.get("result")
            row["raw_result_path"] = str(raw_path) if raw_path.is_file() else str(handle.session_root / "events.jsonl")
            run = None
            if (isinstance(raw, dict) and raw.get("case_id") == case["case_id"]
                    and raw.get("profiling_mode") == internal and raw.get("phase") == ("all" if public is None else "correctness")
                    and isinstance(raw.get("runs"), list) and len(raw["runs"]) == 1):
                run = raw["runs"][0]
                if isinstance(run, dict) and run.get("fixture_id") == "fixture_000" and type(run.get("passed")) is bool:
                    row["correctness"] = "passed" if run["passed"] else "failed"
            # The event includes the CLI's full postcheck; independently check
            # protected records/context/Snapshots without repeating tensor hashes.
            manifest = session._read(handle.protection_root / "manifest.json")
            groups = {r["group"] for r in manifest["roots"].values()} & {"records", "context", "snapshots", "action_context"}
            with session._session_lock(handle.session_root):
                intact = integrity.check_only(handle.protection_root, groups)
            row["integrity_passed"] = event["integrity_passed"] is True and intact.passed
            row["source_hash"] = event.get("content_sha256") or case.get("source_hash")
            artifact_root = action_dir / "workspace" / ".benchmark_outputs" / internal
            row["artifact_check"] = acceptance_artifacts.inspect_artifacts(artifact_root, item)
            row["evaluator_errors"] = ([r.get("error") for r in raw.get("runs", [])
                if isinstance(r, dict) and r.get("error")] if isinstance(raw, dict) and isinstance(raw.get("runs"), list) else [])
            row["message"] = value.get("message")
            if process["timed_out"] or process["cleanup_error"]:
                return row
            stderr = Path(process["stderr_path"]).read_text(errors="replace")
            explicit = unsupported_evidence(raw, public, stderr) if public is not None else None
            if explicit and row["integrity_passed"] and row["correctness"] != "failed" and event.get("error_code") == "action_failed":
                row.update(status="UNSUPPORTED", supported=False, reason="explicit_profiling_unsupported", unsupported_evidence=explicit)
                return row
            if (process["exit_code"] != 0 or value.get("status") != "PASS" or event.get("error_code") is not None
                    or not row["integrity_passed"] or row["correctness"] != "passed"):
                row["reason"] = event.get("error_code") or "action_or_correctness_or_integrity_failed"
                detail = "; ".join(str(e) for e in row["evaluator_errors"])
                if detail:
                    row["reason"] += ":" + detail[:2000]
                return row
            frozen = action_dir / "workspace"
            authority = action_dir / "authority_result.json"
            validated, observation = actions.validate_result(authority, record, frozen,
                "evaluate" if public is None else "profile", internal, process["exit_code"])
            if event.get("content_sha256") != case["source_hash"] or event.get("snapshot_id") != value.get("snapshot_id"):
                raise ValueError("source hash or Snapshot mismatch")
            if validated.get("artifact_dir") != str(artifact_root):
                raise ValueError("artifact root does not belong to this frozen action")
            report = row["artifact_check"]
            row.update(artifact_check=report, raw_result_path=str(authority), latency_us=observation["latency_us"],
                       speedup=run.get("speedup"), snapshot_id=value.get("snapshot_id"))
            if not report["passed"]:
                row["reason"] = "artifact_missing_or_invalid"
                return row
            row.update(status="PASS", supported=True if public is not None else None, reason=None)
        except (OSError, ValueError, TypeError, KeyError, integrity.InfrastructureError, session.SessionError) as error:
            row["reason"] = f"authority_or_artifact_validation:{error}"
        return row

    def run_case(self, case, index):
        if case["identity_errors"]:
            self.blocked_case(case, "; ".join(case["identity_errors"]))
            return
        directory = self.output / "evidence" / f"case_{index:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            _, _, value, environment = session.preflight_case(case["case_id"])
            case["environment"] = environment
            key = digest(environment)
            if key not in self.device_cache:
                self.device_cache[key] = self.device_probe(environment, self.output / "device_checks" / key)
            if self.device_cache[key]:
                self.blocked_case(case, "device_or_toolchain_unavailable:" + "; ".join(self.device_cache[key]))
                return
            case_dir = self.bench / Path(case["case_path"]).parent
            handle = session.create_session(case["case_id"],
                {"id": "acceptance", "backend": "internal_validation", "profile": "full_v02"},
                {"runs_root": self.output / "runs", "snapshots_root": self.output / "snapshots"},
                {"max_hardware_actions": 4, "finish_actions": 1})
            # Session initialization adds the editable default tuning.json to
            # the seed Workspace. Freeze the hash from that published root so
            # fork verification covers the same source tree as the workers.
            case["source_hash"] = workspace.source_manifest(handle.initial_workspace)["content_sha256"]
            protection = session._read(handle.protection_root / "manifest.json")
            for name, global_name in (("case", "cases"), ("sealed", "sealed"), ("environment", "environments")):
                info = protection["roots"][name]
                prefix = Path(info["path"]).relative_to(self.asset_roots[global_name]).as_posix()
                frozen = self.frozen_assets["roots"][global_name]
                for relative, entry in info["entries"].items():
                    actual = {k: entry[k] for k in ("kind", "sha256", "size") if k in entry}
                    key = f"{prefix}/{relative}" if relative else prefix
                    if frozen.get(key) != actual:
                        session.fail_session(handle.session_root, "assets_changed_since_batch_freeze")
                        raise ValueError(f"assets changed since inventory freeze: {global_name}/{key}")
        except (OSError, ValueError, ImportError, RuntimeError) as error:
            self.blocked_case(case, f"case_preflight_or_session_creation:{error}")
            return
        try:
            workers = [handle.initial_workspace]
            for number in range(1, 4):
                fork = self.cli(handle, ["fork"], handle.initial_workspace, directory / f"fork_{number}")
                value = fork["value"]
                if fork["timed_out"] or fork["cleanup_error"] or fork["exit_code"] or not isinstance(value, dict) or value.get("status") != "PASS":
                    raise ValueError("workspace fork failed")
                child = Path(value["workspace"])
                _, child, _ = session.resolve_workspace(handle.session_root, child)
                if workspace.source_manifest(child)["content_sha256"] != case["source_hash"]:
                    raise ValueError("fork seed hash mismatch")
                workers.append(child)
            initial = session.read_session(handle.session_root)
            save(directory / "session_setup.json", {"session_root": str(handle.session_root),
                "workers": [str(p) for p in workers], "source_hash": case["source_hash"], "profiling_modes": initial["profiling_modes"]})
            closed_reason = None
            for (item, public, internal), target in zip(ITEMS, workers):
                if self.stop_reason or closed_reason:
                    row = self.base_row(case, item, public, internal)
                    row["reason"] = self.stop_reason or closed_reason
                elif public is not None and public not in initial["profiling_modes"]:
                    row = self.base_row(case, item, public, internal)
                    row["reason"] = "mode_disabled"
                else:
                    args = ["evaluate"] if public is None else ["profile", public]
                    process = self.cli(handle, args, target, directory / item)
                    row = self.inspect_action(case, handle, item, public, internal, process)
                    record = session.read_session(handle.session_root)
                    if record["status"] == "failed" or record["exploration_closed"]:
                        closed_reason = "session_closed_after_infrastructure_or_interruption"
                self.append(row)
        except Exception as error:
            done = {r["item"] for r in self.rows if r["case_path"] == case["case_path"]}
            for item, public, internal in ITEMS:
                if item not in done:
                    row = self.base_row(case, item, public, internal)
                    row.update(reason=f"case_execution_infrastructure:{error}", session_root=str(handle.session_root))
                    self.append(row)
        finally:
            if not self.stop_reason:
                try:
                    session.close_exploration(handle.session_root, "full_v02_acceptance")
                    environment = os.environ.copy()
                    environment["PYTHONDONTWRITEBYTECODE"] = "1"
                    final = self.run_process([sys.executable, "-c", FINALIZE_CODE, str(self.bench / "evaluator"),
                        str(handle.session_root)], handle.session_root, environment, directory / "finalize", handle.session_root)
                    self.finals.append({"case_path": case["case_path"], "case_id": case["case_id"], **final})
                    save(self.output / "finalizations.json", self.finals)
                except Exception as error:
                    self.finals.append({"case_path": case["case_path"], "error": str(error)})
                    save(self.output / "finalizations.json", self.finals)

    def summarize(self, canonical_check, code_check):
        keys = Counter((row["case_path"], row["item"]) for row in self.rows)
        missing = [(case["case_path"], item) for case in self.cases for item, _, _ in ITEMS
                   if keys[(case["case_path"], item)] != 1]
        complete = not missing and len(self.rows) == len(self.cases)*4
        all_pass = bool(self.cases) and complete and canonical_check["passed"] and code_check["passed"] and not self.stop_reason and all(r["status"] == "PASS" for r in self.rows)
        summary = {"schema_version": 1, "case_count": len(self.cases), "expected_records": len(self.cases)*4,
            "actual_records": len(self.rows), "recording_complete": complete, "missing_or_duplicate": missing,
            "all_pass": all_pass, "canonical_integrity": canonical_check, "public_code_integrity": code_check, "stop_reason": self.stop_reason,
            "by_item": {item: {status: sum(r["item"] == item and r["status"] == status for r in self.rows)
                                for status in STATUSES} for item, _, _ in ITEMS}, "completed_at": now()}
        save(self.output / "summary.json", summary)
        fields = ["case_id", "case_path", "certification", "environment"]
        fields += [f"{item}_{k}" for item, _, _ in ITEMS for k in ("status", "supported", "correctness", "reason", "evidence")]
        with (self.output / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for case in self.cases:
                value = {k: case[k] for k in ("case_id", "case_path", "certification")}
                value["environment"] = case["environment"].get("name")
                for item, _, _ in ITEMS:
                    rows = [r for r in self.rows if r["case_path"] == case["case_path"] and r["item"] == item]
                    row = rows[0] if len(rows) == 1 else {}
                    for key in ("status", "supported", "correctness", "reason"):
                        value[f"{item}_{key}"] = row.get(key)
                    value[f"{item}_evidence"] = row.get("result_path") or "results.jsonl"
                writer.writerow(value)
        lines = ["# S7 全量 Case 验收批次", "", f"Case：{len(self.cases)}；记录：{len(self.rows)}/{len(self.cases)*4}。",
                 f"记录完整：{complete}；全部项目通过：{all_pass}；canonical 内容一致：{canonical_check['passed']}。", "",
                 f"公共代码内容一致：{code_check['passed']}。", "",
                 "| 项目 | PASS | FAIL | BLOCKED | UNSUPPORTED | TIMEOUT |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for item, _, _ in ITEMS:
            lines.append("| " + item + " | " + " | ".join(str(summary["by_item"][item][s]) for s in STATUSES) + " |")
        lines += ["", "## 未通过项目", "", "| Case 路径 | 项目 | 状态 | 原因 | 排查归属 |", "| --- | --- | --- | --- | --- |"]
        for row in self.rows:
            if row["status"] != "PASS":
                reason = str(row["reason"]).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {row['case_path']} | {row['item']} | {row['status']} | {reason} | {row['issue_owner']} |")
        lines += ["", "逐项证据：[results.jsonl](results.jsonl)；全量矩阵：[matrix.csv](matrix.csv)。",
                  "最终复核独立保存在 finalizations.json，不替代上述四项结果。", ""]
        (self.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return summary

    def run(self):
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        session._output({"runs_root": self.output / "runs", "snapshots_root": self.output / "snapshots"})
        self.output.mkdir(parents=True, exist_ok=False)
        for old in (self.bench / "bench", self.bench / "evaluator/bench.py"):
            if os.path.lexists(old):
                raise ValueError("retired CLI still present")
        self.cases = enumerate_cases(self.bench)
        print(f"Freezing {len(self.cases)} Cases and full canonical content hashes...", flush=True)
        before = hash_roots(self.asset_roots)
        self.frozen_assets = before
        code = hash_roots(self.code_roots)
        save(self.output / "canonical_before.json", before)
        save(self.output / "code_before.json", code)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=common.REPO_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
        versions = {}
        for name in ("torch", "torch-npu", "pypto", "numpy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        save(self.output / "inventory.json", {"schema_version": 1, "created_at": now(), "cases": self.cases,
            "revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "python": sys.version, "python_executable": sys.executable, "package_versions": versions, "argv": sys.argv,
            "timeout_seconds": self.timeout, "asset_hash": before["content_sha256"], "code_hash": code["content_sha256"],
            "asset_manifest": "canonical_before.json", "code_manifest": "code_before.json",
            "profiling_open": list(profile_modes.public_enabled_modes()), "hash_errors": before["errors"] + code["errors"]})
        (self.output / "results.jsonl").touch(exist_ok=False)
        save(self.output / "finalizations.json", [])
        if before["errors"] or code["errors"]:
            self.stop_reason = "baseline_hash_failed"
        try:
            for index, case in enumerate(self.cases):
                current_code = hash_roots(self.code_roots)
                if current_code != code:
                    self.stop_reason = "public_code_changed_during_batch"
                if self.stop_reason:
                    self.blocked_case(case, self.stop_reason)
                else:
                    self.run_case(case, index)
        except KeyboardInterrupt:
            self.stop_reason = "batch_interrupted"
            done = {(r["case_path"], r["item"]) for r in self.rows}
            for case in self.cases:
                for item, public, internal in ITEMS:
                    if (case["case_path"], item) not in done:
                        row = self.base_row(case, item, public, internal)
                        row["reason"] = self.stop_reason
                        self.append(row)
        print("Checking complete canonical hashes after acceptance...", flush=True)
        after = hash_roots(self.asset_roots)
        save(self.output / "canonical_after.json", after)
        changes = differences(before, after)
        check = {"passed": not changes and not before["errors"] and not after["errors"],
                 "changed_paths": changes, "errors": before["errors"] + after["errors"]}
        save(self.output / "canonical_comparison.json", check)
        code_after = hash_roots(self.code_roots)
        save(self.output / "code_after.json", code_after)
        code_changes = differences(code, code_after)
        code_check = {"passed": not code_changes and not code["errors"] and not code_after["errors"],
                      "changed_paths": code_changes, "errors": code["errors"] + code_after["errors"]}
        save(self.output / "code_comparison.json", code_check)
        save(self.output / "inventory_after.json", {"cases": enumerate_cases(self.bench)})
        summary = self.summarize(check, code_check)
        print(json.dumps({"output": str(self.output), "recording_complete": summary["recording_complete"],
            "all_pass": summary["all_pass"], "case_count": len(self.cases)}, ensure_ascii=False), flush=True)
        return 0 if summary["all_pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    return AcceptanceRunner(args.out, timeout=args.timeout_seconds).run()


if __name__ == "__main__":
    raise SystemExit(main())
