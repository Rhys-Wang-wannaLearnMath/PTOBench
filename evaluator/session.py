"""Internal Session registry, Linux locks and durable action accounting (S2).

Execution callers hold execution_lock across preparation, reservation, execution
and completion. Status readers acquire only a short record lock. No CLI routing,
Evaluator launch, Snapshot selection or Optimizer lifecycle lives here.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import grp
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import threading
import uuid

import common
import file_tree
import integrity
import profile_modes
import workspace

PUBLIC_MODES = {"runtime": "runtime", "pmu": "hardware", "compiler": "compile",
                "combined": profile_modes.COMBINED_PROFILE_MODE}
UNLIMITED_HARDWARE_ACTIONS = "unlimited"
ERROR_CODES = {"action_failed", "invalid_workspace", "invalid_request", "budget_exceeded",
               "session_closed", "integrity_failed", "infrastructure_error", "interrupted"}
_LOCKS = threading.local()




class SessionError(ValueError):
    def __init__(self, message, error_code="invalid_request"):
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class SessionHandle:
    session_root: Path
    snapshots_root: Path
    protection_root: Path
    initial_workspace: Path


SessionRecord = dict


def context_options(record):
    """Inherit the frozen evaluator policy for generated workspace context."""
    return {'enabled_profiles': list(record['profiling_modes']),
            'profile_strategy': record.get('profile_strategy', profile_modes.DEFAULT_PROFILE_STRATEGY)}


def _now():
    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds(started_at, completed_at):
    if not started_at or not completed_at:
        return None
    return (datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       indent=2) + "\n").encode("utf-8")


def _read(path):
    try:
        with file_tree.open_file(path) as fd, os.fdopen(os.dup(fd), "rb") as stream:
            return json.load(stream)
    except (OSError, ValueError, file_tree.UnsafeTreeError) as error:
        raise integrity.InfrastructureError(f"cannot read {path}: {error}") from error


def _id(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise SessionError(f"invalid {label}")
    return value


def _path(value, *, missing=False, code="invalid_workspace"):
    """Check raw components before normalization (including link/../ tricks)."""
    try:
        value = Path(value)
        raw = value if value.is_absolute() else Path.cwd() / value
        current = Path("/")
        for part in raw.parts[1:]:
            if part == "..":
                current = current.parent
                continue
            current /= part
            try:
                with file_tree.open_directory(current):
                    pass
            except FileNotFoundError:
                if not missing:
                    raise
        return file_tree.absolute_path(raw)
    except (OSError, ValueError, TypeError, file_tree.UnsafeTreeError) as error:
        raise SessionError(f"unsafe directory: {value}: {error}", code) from error


@contextlib.contextmanager
def _flock(path, *, create=False):
    with file_tree.open_directory(path.parent, create=create) as parent:
        fd = os.open(path.name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW |
                     os.O_NONBLOCK | os.O_CLOEXEC, 0o600, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise integrity.InfrastructureError("lock is not a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)


@contextlib.contextmanager
def execution_lock():
    """Fixed benchmark lock; never acquire it while holding a Session lock."""
    if getattr(_LOCKS, "session", False) or getattr(_LOCKS, "execution", None):
        raise integrity.InfrastructureError("lock order violation or recursive execution lock")
    root = _path(common.BENCHMARK_ROOT, code="invalid_request")
    try:
        with _flock(root / ".ptobench/execution.lock", create=True):
            _LOCKS.execution = (os.getpid(), root)
            try:
                yield
            finally:
                _LOCKS.execution = None
    except (OSError, file_tree.UnsafeTreeError) as error:
        raise integrity.InfrastructureError(f"execution lock unavailable: {error}") from error


def _require_execution():
    if getattr(_LOCKS, "execution", None) != (os.getpid(), file_tree.absolute_path(common.BENCHMARK_ROOT)):
        raise integrity.InfrastructureError("operation requires the benchmark execution lock")


@contextlib.contextmanager
def _session_lock(root):
    root = _path(root, code="invalid_request")
    if getattr(_LOCKS, "session", False):
        raise integrity.InfrastructureError("recursive Session lock")
    try:
        with _flock(root / ".session.lock"):
            _LOCKS.session = True
            try:
                yield root
            finally:
                _LOCKS.session = False
    except (OSError, file_tree.UnsafeTreeError) as error:
        raise integrity.InfrastructureError(f"Session lock unavailable: {error}") from error


def _budget(value):
    if value is not None and not isinstance(value, dict):
        raise SessionError("budget must be an object")
    if value and "max_profile_actions" in value:
        raise SessionError("max_profile_actions is no longer supported; evaluate and profile share max_hardware_actions")
    value = {"max_hardware_actions": 20, "finish_actions": 1, **(value or {})}
    maximum = value["max_hardware_actions"]
    valid_maximum = (maximum == UNLIMITED_HARDWARE_ACTIONS or
                     (type(maximum) is int and maximum > 0))
    if (set(value) != {"max_hardware_actions", "finish_actions"}
            or not valid_maximum
            or type(value["finish_actions"]) is not int or value["finish_actions"] != 1):
        raise SessionError("budget requires a valid hardware action limit and exactly one finish action")
    return value


def _asset_file(path):
    with file_tree.open_file(path) as fd:
        if os.fstat(fd).st_size == 0:
            raise SessionError(f"empty asset: {path}")
        return file_tree.descriptor_sha256(fd)


def _inside(base, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or any(p in {"", ".", ".."} for p in relative.split("/"))):
        raise SessionError(f"invalid Case-relative path: {relative!r}")
    return base / relative


def preflight_case(case_id):
    """Check recorded certification and prepared files, without Ref preparation.

    The official builder is called only for spec metadata. No create_tensor,
    golden.run, Ref/Oracle/Candidate entry, or hardware dispatch is called.
    """
    _id(case_id, "Case ID")
    if len(case_id.split(".")) != 2:
        raise SessionError("Case ID must be family.name")
    try:
        case_dir, sealed, case = common.case_paths(case_id)
        case_dir = _path(case_dir, code="invalid_request")
        sealed = _path(sealed, code="invalid_request")
        case = _read(case_dir / "case.json")
        if type(case.get("schema_version")) is not int or case.get("schema_version") != 1 or case.get("case_id") != case_id:
            raise SessionError("Case identity/schema mismatch")
        cert = _read(sealed / "certification.json")
        if (type(cert.get("schema_version")) is not int or cert.get("schema_version") != 1 or cert.get("case_id") != case_id
                or cert.get("status") != "ready" or cert.get("ref_output_cache") != "ready"):
            raise SessionError("Case certification/Ref cache is not ready")
        policy_path = _inside(case_dir, case["comparator"])
        if _asset_file(policy_path) != cert.get("comparator_sha256"):
            raise SessionError("certified comparator hash does not match")
        policy = _read(policy_path)
        outputs = case["correctness"]["outputs"]
        if (not isinstance(outputs, list) or not outputs or len(set(outputs)) != len(outputs)
                or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_]+", n) for n in outputs)
                or type(policy.get("schema_version")) is not int or policy.get("schema_version") != 1 or set(policy.get("outputs", {})) != set(outputs)):
            raise SessionError("invalid comparator/output contract")
        _asset_file(_inside(case_dir, case["config"]))
        seed = _path(_inside(case_dir, case["workspace"]), code="invalid_request")
        _asset_file(_inside(seed, case["module"]))
        workspace.source_manifest(seed)
        for path in (sealed / "evaluate.py", sealed / "ref/ref_npu.py", sealed / "oracle/ref_cpu.py",
                     sealed / "fixture_builder.py"):
            _asset_file(path)
        declared = common.declared_fixture_ids(case)
        performance = _read(sealed / "ref_performance.json")
        if (type(performance.get("schema_version")) is not int or performance.get("schema_version") != 1
                or performance.get("status") != "ready"
                or not isinstance(performance.get("fixtures"), dict)
                or set(performance["fixtures"]) != set(declared)):
            raise SessionError("Ref baseline must cover the exact declared fixture set")
        builder = common.load_isolated_module(sealed / "fixture_builder.py", f"session_specs_{uuid.uuid4().hex}")
        for fixture_id in declared:
            fixture = _path(sealed / "fixtures" / fixture_id, code="invalid_request")
            manifest = _read(fixture / "manifest.json")
            if (type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or manifest.get("fixture_id") != fixture_id
                    or manifest.get("inputs_dir") != "in" or manifest.get("outputs_dir") != "out"):
                raise SessionError("invalid Fixture manifest")
            _path(fixture / "in", code="invalid_request")
            _path(fixture / "out", code="invalid_request")
            # Existing official metadata provides the entire input ABI, including
            # scalar tensors and initial mutating output state. Never regenerate it.
            specs = builder.build(case_dir, sealed, case, manifest["seed"])
            initial = set(case.get("state", {}).get("initial_state", []))
            required = set()
            names = set()
            for spec in specs:
                spec_name = spec.name
                if not isinstance(spec_name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", spec_name) or spec_name in names:
                    raise SessionError("invalid or duplicate official spec name")
                names.add(spec_name)
                if (hasattr(spec, "value") and not hasattr(spec, "shape")) or spec_name not in outputs or spec_name in initial:
                    required.add(spec_name)
            if not required or not set(outputs) <= names or not initial <= required:
                raise SessionError("incomplete official Fixture contract")
            for spec_name in sorted(required):
                _asset_file(fixture / "in" / f"{spec_name}.pt")
            for output_name in outputs:
                _asset_file(fixture / "out" / f"{output_name}.pt")
            median = performance["fixtures"][fixture_id].get("median_effective_us")
            if (type(median) not in (int, float) or not math.isfinite(median) or median <= 0):
                raise SessionError("Ref baseline must be finite and positive for every fixture")
        if performance.get("compiler_tuning") != common.tuning_defaults(case):
            raise SessionError(
                "Ref baseline must explicitly record PYPTO + false and match the complete Case tuning defaults; "
                "complete default baseline verification before opening an Agent Session"
            )
        env_name = _id(case["environment"], "environment")
        environment = _read(common.BENCHMARK_ROOT / "environments" / f"{env_name}.json")
        if environment.get("name") != env_name or not isinstance(environment.get("platform"), str):
            raise SessionError("invalid environment profile")
        return case_dir, sealed, case, environment
    except SessionError:
        raise
    except (OSError, KeyError, TypeError, ValueError, AttributeError, ImportError, RuntimeError,
            file_tree.UnsafeTreeError, integrity.InfrastructureError) as error:
        raise SessionError(f"Case assets unavailable: {error}") from error


def _fixed_roots(case_dir, sealed, case):
    root = common.BENCHMARK_ROOT
    declarations = {
        "case": integrity.ProtectionRoot(case_dir, "fixed"),
        "sealed": integrity.ProtectionRoot(sealed, "fixed"),
        "environment": integrity.ProtectionRoot(root / "environments" / f"{case['environment']}.json", "fixed"),
        "evaluator": integrity.ProtectionRoot(root / "evaluator", "fixed"),
        "golden": integrity.ProtectionRoot(common.REPO_ROOT / "golden", "fixed"),
    }
    with file_tree.open_directory(sealed.parent) as fd:
        for name in sorted(os.listdir(fd)):
            if name.endswith(".py"):
                declarations[f"family_{len(declarations)}"] = integrity.ProtectionRoot(sealed.parent / name, "fixed")
    for name in ("launcher", "ptobench", "pto-launcher"):
        path = root / name
        with file_tree.open_directory(root) as fd:
            try:
                os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
        declarations[f"public_{name}"] = integrity.ProtectionRoot(path, "fixed")
    modes = root / "modes"
    with file_tree.open_directory(root) as fd:
        try:
            os.stat("modes", dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            with file_tree.open_directory(modes) as mode_fd:
                for name in ("__init__.py", *common.MODE_SOURCE_DIRS):
                    try:
                        os.stat(name, dir_fd=mode_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    logical = ("public_modes_package" if name == "__init__.py" else
                               f"public_modes_{name.lstrip('_')}")
                    declarations[logical] = integrity.ProtectionRoot(modes / name, "fixed")
    return declarations


def _output(output):
    if output is None:
        output = {}
    if not isinstance(output, dict) or set(output) - {"runs_root", "snapshots_root"}:
        raise SessionError("output accepts runs_root and snapshots_root")
    bench = file_tree.absolute_path(common.BENCHMARK_ROOT)
    runs = _path(output.get("runs_root", bench / "runs"), missing=True, code="invalid_request")
    snapshots = _path(output.get("snapshots_root", bench / "snapshots"), missing=True, code="invalid_request")
    forbidden = [bench / p for p in ("cases", "environments", "evaluator", "launcher", "modes", ".ptobench")]
    forbidden.append(file_tree.absolute_path(common.sealed_root()))
    forbidden.append(file_tree.absolute_path(common.REPO_ROOT / "golden"))
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a in (runs, snapshots) for b in forbidden):
        raise SessionError("output overlaps canonical assets or public/control code")
    if runs.is_relative_to(snapshots) or snapshots.is_relative_to(runs):
        raise SessionError("runs_root and snapshots_root must not overlap")
    return runs, snapshots


def _validate_agent_traversal(runs_root: Path) -> None:
    """Require the configured Agent identity to traverse the existing root chain."""
    try:
        import agent_launch
        agent_uid, share_gid = agent_launch._agent_identity()
        current = Path('/')
        for part in runs_root.parts[1:]:
            current /= part
            try:
                current.lstat()
            except FileNotFoundError:
                raise SessionError(
                    f"runs_root must be preconfigured and traversable by ptobench-agent: {current}",
                    'invalid_request') from None
            if not current.is_dir():
                raise SessionError(f"runs_root path component is not a directory: {current}", 'invalid_request')
            permissions = agent_launch._agent_directory_permissions(current, agent_uid, share_gid)
            if 'x' not in permissions:
                raise SessionError(
                    f"Agent cannot traverse runs_root at {current}; an administrator must "
                    "preconfigure traversal for ptobench-agent", 'invalid_request')
    except SessionError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise SessionError(
            f"cannot verify Agent traversal of runs_root {runs_root}; an administrator must "
            f"preconfigure the path for ptobench-agent: {error}", 'invalid_request') from error


def _share_gid() -> int:
    try:
        return grp.getgrnam('ptobench-share').gr_gid
    except KeyError as error:
        raise SessionError('ptobench-share is unavailable; Session workspace cannot be prepared') from error


def _prepare_agent_traversal_directory(path: Path, share_gid: int) -> None:
    try:
        with file_tree.open_directory(path) as descriptor:
            metadata = os.fstat(descriptor)
            if metadata.st_uid != os.getuid():
                raise SessionError(f'Agent traversal directory must be Broker-owned: {path}')
            os.fchown(descriptor, -1, share_gid)
            os.fchmod(descriptor, 0o2710)
    except SessionError:
        raise
    except OSError as error:
        raise SessionError(f'cannot prepare Agent traversal directory {path}: {error}') from error


def _prepare_session_path(root: Path, runs_root: Path, optimizer_id: str, case_id: str) -> None:
    """Create the Broker-owned traversal chain without exposing directory listings."""
    share_gid = _share_gid()
    optimizer_root = runs_root / optimizer_id
    case_root = optimizer_root / case_id
    for directory in (optimizer_root, case_root):
        with file_tree.open_directory(directory, create=True):
            pass
        _prepare_agent_traversal_directory(directory, share_gid)


def create_session(case_id, optimizer, output=None, budget=None, *, mode=None,
                   initial_record_paths=(), profile_strategy=None) -> SessionHandle:
    budget = _budget(budget)
    # The evaluator records an opaque participant identity; it never loads its code,
    # instructions or permissions. All methods use the same evaluator policy.
    method = {'id': _id(mode, 'Method ID')} if mode is not None else None
    profile_strategy = profile_modes.normalize_strategy(profile_strategy)
    public_profiles = list(profile_modes.public_enabled_modes(profile_strategy))
    if isinstance(optimizer, str):
        optimizer = {"backend": optimizer, "profile": "default"}
    if not isinstance(optimizer, dict) or set(optimizer) - {"id", "backend", "profile", "command", "tool_version"}:
        raise SessionError("invalid optimizer configuration")
    optimizer = copy.deepcopy(optimizer)
    optimizer_id = _id(optimizer.get("id", optimizer.get("backend")), "Optimizer ID")
    _id(optimizer.get("backend"), "backend")
    _id(optimizer.setdefault("profile", "default"), "profile")
    if (optimizer.get("command") is not None and (not isinstance(optimizer["command"], list)
            or any(not isinstance(arg, str) for arg in optimizer["command"]))
            or optimizer.get("tool_version") is not None and not isinstance(optimizer["tool_version"], str)):
        raise SessionError("optimizer command must be an argv list; tool_version must be text")
    if (not isinstance(initial_record_paths, (tuple, list))
            or any(not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
                   or ".." in PurePosixPath(name).parts or str(PurePosixPath(name)) != name
                   for name in initial_record_paths)
            or len(set(initial_record_paths)) != len(initial_record_paths)):
        raise SessionError("invalid initial record paths")
    runs, snapshots = _output(output)
    with execution_lock():
        case_dir, sealed, case, environment = preflight_case(case_id)
        _validate_agent_traversal(runs)
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
        root = runs / optimizer_id / case_id / session_id
        snapshot_root = snapshots / optimizer_id / case_id / session_id
        created = []
        publishing = False
        try:
            _prepare_session_path(root, runs, optimizer_id, case_id)
            for target in (root, snapshot_root):
                with file_tree.open_directory(target.parent, create=True) as fd:
                    os.mkdir(target.name, mode=0o700, dir_fd=fd)
                created.append(target)
                if target == root:
                    _prepare_agent_traversal_directory(root, _share_gid())
            for name in ("workspaces", "staging", "final", "protection"):
                with file_tree.open_directory(root / name, create=True):
                    pass
                if name == 'workspaces':
                    with file_tree.open_directory(root / name) as fd:
                        os.fchown(fd, -1, _share_gid())
                        os.fchmod(fd, 0o2750)
                else:
                    with file_tree.open_directory(root / name) as fd:
                        os.fchmod(fd, 0o700)
            with file_tree.open_directory(root / 'staging/forks', create=True) as fd:
                os.fchmod(fd, 0o700)
            file_tree.atomic_bytes(root / "process.log", b"")
            initial = root / "workspaces/worker_000"
            initial_stage = root / 'staging/forks/worker_000'
            with file_tree.open_directory(initial_stage, create=True) as fd:
                os.fchmod(fd, 0o700)
            workspace.init_workspace(case_dir, sealed, case, initial_stage, publish=False,
                                     workspace_path=initial,
                                     enabled_profiles=public_profiles,
                                     profile_strategy=profile_strategy)
            if os.stat(initial_stage).st_dev != os.stat(root / 'workspaces').st_dev:
                raise SessionError('Workspace staging and workspaces must share a filesystem')
            with file_tree.open_directory(initial_stage.parent) as source_fd, \
                    file_tree.open_directory(initial.parent) as destination_fd:
                os.rename(initial_stage.name, initial.name,
                          src_dir_fd=source_fd, dst_dir_fd=destination_fd)
            with _session_lock(root):
                declarations = _fixed_roots(case_dir, sealed, case)
                declarations.update({name: integrity.ProtectionRoot(root / filename, "records")
                                     for name, filename in (("session", "session.json"), ("events", "events.jsonl"), ("final_result", "final_result.json"))})
                declarations["context_worker_000"] = integrity.ProtectionRoot(
                    initial / ".benchmark_context", "context", readonly=False)
                for name in initial_record_paths:
                    path = root / name
                    with file_tree.open_directory(path.parent, create=True):
                        pass
                    logical = "launcher_" + name.replace(".", "_").replace("/", "_")
                    if logical in declarations:
                        raise SessionError("initial record path conflicts with Session state")
                    declarations[logical] = integrity.ProtectionRoot(path, "records")
                manifest = integrity.capture_protection(root / "protection", declarations)
                workspace.publish_workspace_root(initial)
            if workspace.validate_workspace(initial, root_mode=0o3770,
                    workspace_path=initial, case_id=case_id) is None:
                raise workspace.InvalidWorkspaceError('published initial Workspace is incomplete')
            with _session_lock(root):
                record = {
                    "schema_version": 2, "session_id": session_id, "optimizer_id": optimizer_id,
                    "case_id": case_id, "status": "running", "exploration_closed": False, "close_reason": None,
                    "budget": budget, "usage": {"hardware_actions": 0, "finish_actions": 0},
                    "workspaces": {"worker_000": {"path": "workspaces/worker_000", "parent": None}},
                    "pending_forks": {},
                    "next_workspace_index": 1, "next_event_seq": 1, "active_action": None,
                    "snapshots_root": str(snapshot_root), "runs_root": str(runs), "session_root": str(root),
                    "benchmark_root": str(file_tree.absolute_path(common.BENCHMARK_ROOT)),
                    "case_dir": str(case_dir), "sealed_dir": str(sealed), "optimizer": optimizer,
                    "created_at": _now(), "completed_at": None, "environment": environment,
                    "mode": method, "profile_strategy": profile_strategy,
                    "profiling_modes": public_profiles,
                    "command": optimizer.get("command"), "tool_version": optimizer.get("tool_version"),
                    "bench_python": sys.version.split()[0], "asset_hashes": {
                        name: hashlib.sha256(_bytes(info["entries"])).hexdigest()
                        for name, info in manifest["roots"].items() if info["group"] == "fixed"},
                }
                _validate_record(root, record)
                payload = _bytes(record)
                publishing = True
                integrity.commit_records(root / "protection", {"session": payload, "events": b""})
            return SessionHandle(root, snapshot_root, root / "protection", initial)
        except BaseException:
            # A manifest-published record may already be durable. Leave it for
            # normal S1 recovery rather than deleting a possibly live Session.
            if not publishing:
                for target in reversed(created):
                    with file_tree.open_parent(target) as (fd, name):
                        file_tree.remove_at(fd, name)
            raise


def _validate_record(root, record):
    try:
        if not isinstance(record, dict) or type(record["schema_version"]) is not int or record["schema_version"] != 2:
            raise ValueError("schema")
        for name in ("session_id", "optimizer_id", "case_id"):
            _id(record[name], name)
        if not re.fullmatch(r"\d{8}T\d{6}Z_[a-f0-9]{12}", record["session_id"]):
            raise ValueError("Session ID")
        if len(record["case_id"].split(".")) != 2:
            raise ValueError("Case ID")
        _id(record["optimizer"]["backend"], "backend")
        _id(record["optimizer"]["profile"], "profile")
        if record["optimizer"].get("id", record["optimizer"]["backend"]) != record["optimizer_id"]:
            raise ValueError("Optimizer identity")
        if (not isinstance(record["created_at"], str) or not isinstance(record["environment"], dict)
                or not isinstance(record["asset_hashes"], dict)
                or not {"case", "sealed", "environment", "evaluator", "golden"} <= record["asset_hashes"].keys()
                or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in record["asset_hashes"].values())):
            raise ValueError("provenance")
        if (record["session_root"] != str(root) or record["benchmark_root"] != str(file_tree.absolute_path(common.BENCHMARK_ROOT))
                or root != Path(record["runs_root"]) / record["optimizer_id"] / record["case_id"] / record["session_id"]):
            raise ValueError("frozen location")
        expected_case, expected_sealed, _ = common.case_paths(record["case_id"])
        if record["case_dir"] != str(expected_case) or record["sealed_dir"] != str(expected_sealed):
            raise ValueError("Case identity")
        if not Path(record["snapshots_root"]).is_absolute() or Path(record["snapshots_root"]).parts[-3:] != (record["optimizer_id"], record["case_id"], record["session_id"]):
            raise ValueError("Snapshot location")
        if record["status"] not in {"running", "exhausted", "completed", "failed"} or type(record["exploration_closed"]) is not bool:
            raise ValueError("state")
        _budget(record["budget"])
        if set(record["usage"]) != {"hardware_actions", "finish_actions"}:
            raise ValueError("usage")
        for name, maximum in (("hardware_actions", None), ("finish_actions", 1)):
            used = record["usage"][name]
            if type(used) is not int or used < 0 or (maximum is not None and used > maximum):
                raise ValueError("usage limits")
        if record["status"] in {"failed", "completed"} and not record["exploration_closed"]:
            raise ValueError("terminal exploration")
        if (record["exploration_closed"] and (not isinstance(record["close_reason"], str) or not record["close_reason"])) or (not record["exploration_closed"] and record["close_reason"] is not None):
            raise ValueError("close reason")
        if type(record["next_workspace_index"]) is not int or record["next_workspace_index"] < 1 or type(record["next_event_seq"]) is not int or record["next_event_seq"] < 1:
            raise ValueError("counters")
        if not isinstance(record["workspaces"], dict) or "worker_000" not in record["workspaces"]:
            raise ValueError("registry")
        for name, info in record["workspaces"].items():
            if not re.fullmatch(r"worker_\d{3,}", name) or int(name[7:]) >= record["next_workspace_index"] or set(info) != {"path", "parent"} or info["path"] != f"workspaces/{name}":
                raise ValueError("workspace binding")
            if name == "worker_000":
                if info["parent"] is not None:
                    raise ValueError("initial parent")
            elif info["parent"] not in record["workspaces"] or int(info["parent"][7:]) >= int(name[7:]):
                raise ValueError("workspace parent")
        pending = record["pending_forks"]
        if not isinstance(pending, dict):
            raise ValueError('pending forks')
        staging_paths = set()
        for name, info in pending.items():
            if (not re.fullmatch(r"worker_\d{3,}", name)
                    or int(name[7:]) >= record['next_workspace_index']
                    or not isinstance(info, dict) or set(info) != {'worker_id', 'parent', 'staging', 'phase'}
                    or info['worker_id'] != name or info['parent'] not in record['workspaces']
                    or name in record['workspaces']
                    and record['workspaces'][name]['parent'] != info['parent']
                    or int(info['parent'][7:]) >= int(name[7:])
                    or info['staging'] != f'staging/forks/{name}'
                    or info['phase'] not in {'reservation', 'staged', 'renamed-locked',
                                             'context-registered', 'registered-locked',
                                             'registered-open'}
                    or info['staging'] in staging_paths):
                raise ValueError('pending fork binding')
            staging_paths.add(info['staging'])
        if (not isinstance(record["profiling_modes"], list) or len(set(record["profiling_modes"])) != len(record["profiling_modes"])
                or not set(record["profiling_modes"]) <= PUBLIC_MODES.keys()):
            raise ValueError("public modes")
        profile_modes.normalize_strategy(record.get("profile_strategy"))
        method = record.get('mode')
        if method is not None:
            if not isinstance(method, dict):
                raise ValueError('method identity')
            _id(method.get('id'), 'Method ID')
        active = record["active_action"]
        if active is not None:
            if (not isinstance(active, dict) or type(active["event_seq"]) is not int
                    or not 0 < active["event_seq"] < record["next_event_seq"]
                    or active["kind"] not in {"evaluate", "profile", "finish"}
                    or not re.fullmatch(r"[a-f0-9]{64}", active["content_sha256"])
                    or active["workspace"] is None or not isinstance(active["started_at"], str)):
                raise ValueError("active action")
            if (active["process_state"] not in {"not_started", "starting", "running"}
                    or (active["process_state"] == "running") != (active["process"] is not None)
                    or active["owner"] is None
                    or "observed_mode_changes" in active
                    and (not isinstance(active["observed_mode_changes"], list)
                         or any(not isinstance(path, str) for path in active["observed_mode_changes"]))
                    or "attempt_id" in active and (type(active["attempt_id"]) is not int or active["attempt_id"] < 1)
                    or "compile_config" in active and active["compile_config"] is not None
                    and not isinstance(active["compile_config"], dict)
                    or "parent_attempt_id" in active and active["parent_attempt_id"] is not None
                    and (type(active["parent_attempt_id"]) is not int or active["parent_attempt_id"] < 1)):
                raise ValueError("action process state")
            if active["kind"] == "finish":
                if active["workspace"] != "final" or not record["exploration_closed"] or record["usage"]["finish_actions"] != 1:
                    raise ValueError("finish binding")
            elif active["workspace"] not in record["workspaces"] or record["usage"]["hardware_actions"] == 0:
                raise ValueError("exploration binding")
            if (active["kind"] == "profile" and active["mode"] not in record["profiling_modes"]
                    or active["kind"] != "profile" and active["mode"] is not None):
                raise ValueError("action mode")
            for identity in (active["owner"], active["process"]):
                if identity is not None and (set(identity) != {"pid", "start_ticks", "pgid"} or any(type(v) is not int or v <= 0 for v in identity.values())):
                    raise ValueError("process identity")
    except (KeyError, TypeError, ValueError, SessionError) as error:
        raise integrity.InfrastructureError(f"invalid Session schema: {error}") from error


def _load_locked(root, repair=True, *, allow_pending=False):
    protection = root / "protection"
    # Inspect retired Workspace markers and the hash-verified Session backup
    # before any automatic record restoration can rewrite a historical Session.
    manifest = integrity._load(protection)
    _reject_retired_workspace_protocol(root, manifest)
    # S1 validates the complete manifest, even when inspecting one group.
    report = integrity.check_only(protection, "records")
    for name, info in manifest["roots"].items():
        if info["group"] == "records" and not Path(info["path"]).is_relative_to(root):
            raise integrity.InfrastructureError("record recovery path is outside this Session")
        if info["group"] == "context":
            if re.fullmatch(r"context_worker_\d{3,}", name):
                expected = root / "workspaces" / name[8:] / ".benchmark_context"
            elif name == "context_final":
                expected = root / "final/workspace/.benchmark_context"
            else:
                raise integrity.InfrastructureError("invalid context protection name")
            if info["path"] != str(expected) or info["readonly"]:
                raise integrity.InfrastructureError("context recovery path is outside its workspace")
    for name, filename in (("session", "session.json"), ("events", "events.jsonl"), ("final_result", "final_result.json")):
        info = manifest["roots"].get(name, {})
        if info.get("path") != str(root / filename) or info.get("group") != "records":
            raise integrity.InfrastructureError("manifest does not bind this Session's records")
    if repair:
        restored = integrity.check_and_restore(protection, "records")
        report.changed_paths = sorted(set(report.changed_paths) | set(restored.changed_paths))
    elif not report.passed:
        raise integrity.InfrastructureError("Session records changed", report)
    record = _read(root / "session.json")
    _validate_record(root, record)
    if record['pending_forks'] and not allow_pending:
        raise SessionError('Session has an unfinished Workspace fork; recovery is required', 'invalid_request')
    fixed = {name: info for name, info in manifest["roots"].items() if info["group"] == "fixed"}
    expected_paths = {
        "case": Path(record["case_dir"]), "sealed": Path(record["sealed_dir"]),
        "evaluator": common.BENCHMARK_ROOT / "evaluator", "golden": common.REPO_ROOT / "golden",
        "environment": common.BENCHMARK_ROOT / "environments" / f"{record['environment']['name']}.json",
        "public_ptobench": common.BENCHMARK_ROOT / "ptobench",
        "public_pto-launcher": common.BENCHMARK_ROOT / "pto-launcher",
        "public_launcher": common.BENCHMARK_ROOT / "launcher",
        "public_modes": common.BENCHMARK_ROOT / "modes",
        "public_modes_package": common.BENCHMARK_ROOT / "modes/__init__.py",
        **{f"public_modes_{name.lstrip('_')}": common.BENCHMARK_ROOT / "modes" / name
           for name in common.MODE_SOURCE_DIRS},
    }
    if fixed.keys() != record["asset_hashes"].keys():
        raise integrity.InfrastructureError("fixed asset protection scope changed")
    for name, info in fixed.items():
        path = Path(info["path"])
        if hashlib.sha256(_bytes(info["entries"])).hexdigest() != record["asset_hashes"][name]:
            raise integrity.InfrastructureError("initial fixed asset inventory changed")
        if name in expected_paths:
            valid = path == expected_paths[name]
        elif name.startswith("family_"):
            valid = path.parent == Path(record["sealed_dir"]).parent and path.suffix == ".py"
        else:
            valid = False
        if not valid:
            raise integrity.InfrastructureError("fixed asset path binding changed")
    for name, info in record["workspaces"].items():
        context = manifest["roots"].get(f"context_{name}", {})
        if context.get("path") != str(root / info["path"] / ".benchmark_context") or context.get("group") != "context":
            raise integrity.InfrastructureError("registry context is not protected")
    events = _events(root, record)
    if repair and report.changed_paths and record['active_action'] is not None:
        active = record['active_action']
        actionable = set(report.changed_paths) - set(report.blocked_paths)
        active['integrity_violations'] = sorted(set(active.get('integrity_violations', [])) | actionable)
        _commit(root, record)
    return record, events


_RETIRED_WORKSPACE_ROOTS = frozenset({
    'public_bench', 'candidate_pool', 'candidates', 'public_workspaces',
    'workspace_exports', 'exported_workspaces',
})


def _reject_retired_workspace_protocol(root, manifest):
    roots = manifest['roots']
    if (_RETIRED_WORKSPACE_ROOTS & roots.keys()
            or 'bench.py' in roots.get('evaluator', {}).get('entries', {})):
        raise SessionError('Session uses a retired Workspace protocol; create a fresh Session', 'invalid_request')
    entry = roots.get('session', {}).get('entries', {}).get('')
    if not entry or entry.get('kind') != 'file':
        raise integrity.InfrastructureError('Session record has no protected format backup')
    integrity._verify_backup(root / 'protection', entry)
    try:
        with file_tree.open_file(root / 'protection' / entry['backup']) as descriptor, \
                os.fdopen(os.dup(descriptor), 'rb') as stream:
            backup = json.load(stream)
    except (OSError, ValueError, UnicodeError, file_tree.UnsafeTreeError) as error:
        raise integrity.InfrastructureError(f'cannot read protected Session format backup: {error}') from error
    if (isinstance(backup, dict) and type(backup.get('schema_version')) is int
            and backup['schema_version'] != 2):
        raise SessionError('Session uses a retired Workspace protocol; create a fresh Session', 'invalid_request')


def _events(root, record):
    with file_tree.open_file(root / "events.jsonl") as fd, os.fdopen(os.dup(fd), "rb") as stream:
        try:
            events = [json.loads(line) for line in stream if line.strip()]
        except (ValueError, UnicodeError) as error:
            raise integrity.InfrastructureError("malformed events") from error
    seen = set()
    attempt_sequence = 0
    counters = {"hardware_actions": 0, "finish_actions": 0}
    for event in events:
        seq = event.get("seq") if isinstance(event, dict) else None
        if type(seq) is not int or not 0 < seq < record["next_event_seq"] or seq in seen:
            raise integrity.InfrastructureError("invalid/duplicate event sequence")
        seen.add(seq)
        fields = {"seq", "time", "action", "mode", "workspace", "content_sha256", "exit_code",
                  "error_code", "hardware_action", "action_started", "integrity_passed", "repaired_paths",
                  "result", "snapshot_id", "hardware_actions_used"}
        if (not fields <= event.keys() or event["action"] not in {"evaluate", "profile", "fork", "finish", "invalid_request"}
                or event.get("run_id") is not None and event["run_id"] != record["session_id"]
                or any(type(event[key]) is not bool for key in ("hardware_action", "action_started", "integrity_passed"))
                or type(event["exit_code"]) is not int or event["error_code"] is not None and event["error_code"] not in ERROR_CODES
                or not isinstance(event["repaired_paths"], list)
                or "observed_mode_changes" in event
                and (not isinstance(event["observed_mode_changes"], list)
                     or any(not isinstance(path, str) for path in event["observed_mode_changes"]))
                or type(event["hardware_actions_used"]) is not int
                or not 0 <= event["hardware_actions_used"] <= record["usage"]["hardware_actions"]):
            raise integrity.InfrastructureError("invalid event schema")
        skipped = event.get("candidate_capture_skipped")
        if (skipped is not None and (not isinstance(skipped, list)
                or any(not isinstance(path, str) or not path for path in skipped)
                or skipped != sorted(set(skipped)))):
            raise integrity.InfrastructureError("invalid Candidate capture diagnostics")
        snapshot_error = event.get("snapshot_error")
        if (snapshot_error is not None and
                (not isinstance(snapshot_error, str) or not snapshot_error
                 or event["action"] not in {"evaluate", "profile"}
                 or event["snapshot_id"] is not None or event["error_code"] is not None
                 or event["exit_code"] != 0 or not event["hardware_action"]
                 or not event["action_started"] or not event["integrity_passed"])):
            raise integrity.InfrastructureError("invalid optional Snapshot failure evidence")
        if event["hardware_action"]:
            if event["action"] in {"fork", "invalid_request"} or not isinstance(event["content_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", event["content_sha256"]):
                raise integrity.InfrastructureError("invalid accepted event")
            counters["finish_actions" if event["action"] == "finish" else "hardware_actions"] += 1
        if "attempt_id" in event and (type(event["attempt_id"]) is not int or event["attempt_id"] < 1):
            raise integrity.InfrastructureError("invalid Attempt number")
        if event.get("parent_attempt_id") is not None and (
                type(event["parent_attempt_id"]) is not int or event["parent_attempt_id"] < 1):
            raise integrity.InfrastructureError("invalid parent Attempt")
        if event["action"] in {"evaluate", "profile", "finish"}:
            attempt_sequence += 1
            if "attempt_id" in event and event["attempt_id"] != attempt_sequence:
                raise integrity.InfrastructureError("Attempt order does not match the event timeline")
    active = record["active_action"]
    if active and active["event_seq"] in seen:
        raise integrity.InfrastructureError("active action already has a completed event")
    if active:
        seen.add(active["event_seq"])
        counters["finish_actions" if active["kind"] == "finish" else "hardware_actions"] += 1
    if seen != set(range(1, record["next_event_seq"])) or counters != record["usage"]:
        raise integrity.InfrastructureError("event/active sequence or budget ledger mismatch")
    maximum = record["budget"]["max_hardware_actions"]
    if maximum != UNLIMITED_HARDWARE_ACTIONS and counters["hardware_actions"] > maximum:
        raise integrity.InfrastructureError("hardware events exceed the Session allowance")
    return events


def read_session(session_root, repair=True, *, allow_pending=False) -> SessionRecord:
    with _session_lock(session_root) as root:
        return copy.deepcopy(_load_locked(root, repair, allow_pending=allow_pending)[0])


def _commit(root, record, events=None, record_updates=None):
    _validate_record(root, record)
    replacements = {"session": _bytes(record)}
    if events is not None:
        replacements["events"] = b"".join((json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for event in events)
    if record_updates:
        if {"session", "events"} & record_updates.keys():
            raise SessionError("extra records cannot replace Session or events")
        replacements.update(record_updates)
    integrity.commit_records(root / "protection", replacements)


def _open_exploration(record):
    if record["status"] == "exhausted":
        raise SessionError("Hardware action budget exhausted; exploration actions are closed", "budget_exceeded")
    if record["status"] in {"failed", "completed"} or record["exploration_closed"]:
        raise SessionError("Session exploration is closed", "session_closed")
    if record["active_action"] is not None:
        raise SessionError("another accepted action still needs completion")


def _workspace_path(root, record, value):
    target = _path(value)
    if not target.is_relative_to(root / "workspaces"):
        raise SessionError("workspace is outside this Session", "invalid_workspace")
    for name, info in record["workspaces"].items():
        if target == root / info["path"]:
            return name, target
    raise SessionError("workspace must be a registered root", "invalid_workspace")


def resolve_workspace(session_root, value):
    """Explicit Session root only. Public metadata never determines identity."""
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root)
        name, target = _workspace_path(root, record, value)
        return name, target, record["case_id"]


def require_current_evaluator(manifest):
    # Older Sessions remain readable as historical evidence, but their fixed
    # backups must not reinstall the former Agent profile validation policy.
    policy = manifest['roots'].get('evaluator', {}).get('entries', {}).get('profile_execution.py', {})
    if ((common.BENCHMARK_ROOT / 'evaluator/profile_execution.py').is_file()
            and policy.get('sha256') != '7fe254fb7c2cb0c870f67f4dc6f2dc9f8511f044a8b242899a29c442310ea298'):
        raise integrity.InfrastructureError('Session captured the older profiling policy; create a fresh Session')
    runtime_policy = manifest['roots'].get('evaluator', {}).get('entries', {}).get('runtime_policy.py', {})
    if ((common.BENCHMARK_ROOT / 'evaluator/runtime_policy.py').is_file()
            and runtime_policy.get('sha256') != '029e1ff84fc786d023088b1a1111a7396bc91c7e0dd88a35c72864e0eed7cd21'):
        raise integrity.InfrastructureError(
            'Session captured the older runtime environment policy; create a fresh Session'
        )


    tuning_path = common.BENCHMARK_ROOT / 'evaluator/tuning.py'
    tuning_policy = manifest['roots'].get('evaluator', {}).get('entries', {}).get('tuning.py', {})
    if tuning_path.is_file() and tuning_policy.get('sha256') != common.sha256_file(tuning_path):
        raise integrity.InfrastructureError(
            'Session captured a different tuning policy; create a fresh Session'
        )
    registry = common.BENCHMARK_ROOT / 'evaluator/tuning_policies.json'
    captured = manifest['roots'].get('evaluator', {}).get('entries', {}).get('tuning_policies.json', {})
    if registry.is_file() and captured.get('sha256') != common.sha256_file(registry):
        raise integrity.InfrastructureError('Session captured different Case tuning permissions; create a fresh Session')


def _fail_integrity_session(session_root, reason):
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root)
        record.update(status="failed", exploration_closed=True,
                      close_reason=reason, completed_at=_now())
        _commit(root, record)


def repair_assets(session_root):
    _require_execution()
    with _session_lock(session_root) as root:
        _load_locked(root)
        require_current_evaluator(_read(root / 'protection/manifest.json'))
    try:
        report = integrity.check_and_restore(root / "protection", {"fixed", "context"})
        manifest = integrity._load(root / 'protection')
        for protected in manifest['roots'].values():
            if protected['group'] == 'context':
                try:
                    workspace._readonly(Path(protected['path']))
                except (KeyError, OSError, ValueError, RuntimeError) as error:
                    raise integrity.InfrastructureError(
                        f'cannot normalize recovered Context {protected["path"]}: {error}') from error
        return report
    except integrity.InfrastructureError:
        _fail_integrity_session(root, "asset_recovery_failed")
        raise


def register_workspace(session_root, workspace_path, parent_id=None):
    """Finish registration only for a durable pending reservation."""
    _require_execution()
    target = _path(workspace_path)
    name = target.name
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root, allow_pending=True)
        _open_exploration(record)
        reservation = record['pending_forks'].get(name)
        if (target.parent != root / 'workspaces' or reservation is None
                or reservation['parent'] != parent_id or name in record['workspaces']):
            raise SessionError('not the allocated workspace or parent', 'invalid_workspace')
    recover_pending_forks(root)
    return name


def fork_workspace(session_root, parent_workspace):
    """Allocate a child with the global lock held across the operation."""
    with execution_lock():
        return fork_workspace_locked(session_root, parent_workspace)


def fork_workspace_locked(session_root, parent_workspace):
    """Shared internal path for a caller already holding the execution lock."""
    _require_execution()
    recover_pending_forks(session_root)
    repair_assets(session_root)
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root)
        _open_exploration(record)
        parent_id, parent = _workspace_path(root, record, parent_workspace)
        index = record["next_workspace_index"]
        name = f"worker_{index:03d}"
        target = root / "workspaces" / name
        stage = root / 'staging/forks' / name
        with file_tree.open_directory(target.parent) as fd:
            try:
                os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise SessionError("allocated child already exists", "invalid_workspace")
        with file_tree.open_directory(stage.parent) as fd:
            try:
                os.stat(stage.name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise SessionError('fork staging path already exists', 'invalid_workspace')
        record["next_workspace_index"] += 1
        record['pending_forks'][name] = {
            'worker_id': name, 'parent': parent_id,
            'staging': f'staging/forks/{name}', 'phase': 'reservation',
        }
        _commit(root, record)
    try:
        with file_tree.open_directory(stage, create=True) as fd:
            os.fchown(fd, -1, _share_gid())
            os.fchmod(fd, 0o700)
        workspace.copy_candidate_tree(parent, stage / 'candidate')
        case_dir, sealed = Path(record["case_dir"]), Path(record["sealed_dir"])
        case = _read(case_dir / "case.json")
        workspace.write_context(case_dir, sealed, case, stage,
                                parent_workspace=parent, workspace_path=target, **context_options(record))
        workspace._ensure_tuning(stage, case)
        workspace._normalize_workspace(stage, publish=False)
        if os.stat(stage).st_dev != os.stat(target.parent).st_dev:
            raise SessionError('Workspace staging and workspaces must share a filesystem')
        if workspace.validate_workspace(stage, root_mode=0o700, workspace_path=target,
                                        parent_workspace=parent, case_id=record['case_id']) is None:
            raise workspace.InvalidWorkspaceError('Fork staging tree is incomplete')
        _set_fork_phase(root, name, 'staged')
        _rename_fork_stage(stage, target)
        _set_fork_phase(root, name, 'renamed-locked')
        recover_pending_forks(root)
        return name
    except BaseException:
        _cancel_unpublished_reservation(root, name)
        raise


def _set_fork_phase(root, name, phase):
    with _session_lock(root):
        record, _ = _load_locked(root, allow_pending=True)
        pending = record['pending_forks'].get(name)
        if pending is None:
            raise integrity.InfrastructureError(f'fork reservation disappeared: {name}')
        pending['phase'] = phase
        _commit(root, record)


def _rename_fork_stage(stage, target):
    with file_tree.open_directory(stage.parent) as source_fd, \
            file_tree.open_directory(target.parent) as destination_fd:
        try:
            os.stat(target.name, dir_fd=destination_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise integrity.InfrastructureError(f'fork target already exists: {target}')
        os.rename(stage.name, target.name, src_dir_fd=source_fd, dst_dir_fd=destination_fd)
        os.fsync(source_fd)
        os.fsync(destination_fd)


def _cancel_unpublished_reservation(root, name):
    with _session_lock(root):
        record, _ = _load_locked(root, allow_pending=True)
        pending = record['pending_forks'].get(name)
        if pending is None or pending['phase'] != 'reservation':
            return
        target = root / 'workspaces' / name
        if target.exists() or target.is_symlink():
            return
        stage = root / pending['staging']
        with file_tree.open_directory(stage.parent) as fd:
            file_tree.remove_at(fd, stage.name)
        record['pending_forks'].pop(name)
        _commit(root, record)


def _context_root_registered(root, name, target):
    protection = root / 'protection'
    manifest = integrity._load(protection)
    key = f'context_{name}'
    info = manifest['roots'].get(key)
    if info is None:
        integrity.register_roots(protection, {
            key: integrity.ProtectionRoot(target / '.benchmark_context', 'context', readonly=False),
        })
        return
    if (info['path'] != str(target / '.benchmark_context') or info['group'] != 'context'
            or info['readonly'] is not False):
        raise integrity.InfrastructureError(f'fork Context integrity binding is invalid: {name}')
    report = integrity.check_only(protection, {'context'})
    if not report.passed:
        raise integrity.InfrastructureError(f'fork Context integrity check failed: {name}', report)


def _discard_unpublished_fork_reservation(root, record, name, pending, *, stage=None):
    """Remove a reservation only while neither Workspace nor Context is published."""
    target = root / 'workspaces' / name
    if pending['phase'] != 'reservation' or target.exists() or target.is_symlink():
        raise integrity.InfrastructureError(f'fork reservation has published Workspace state: {name}')
    if name in record['workspaces']:
        raise integrity.InfrastructureError(f'fork reservation has registered Workspace state: {name}')
    manifest = integrity._load(root / 'protection')
    if f'context_{name}' in manifest['roots']:
        raise integrity.InfrastructureError(f'fork reservation has registered Context state: {name}')
    if stage is not None:
        if stage.is_symlink():
            raise integrity.InfrastructureError(f'fork reservation staging path is a symlink: {name}')
        with file_tree.open_directory(stage.parent) as descriptor:
            file_tree.remove_at(descriptor, stage.name)
    record['pending_forks'].pop(name)
    _commit(root, record)


def _finish_pending_fork(root, record, name, pending, *, stage_exists):
    target = root / 'workspaces' / name
    parent_id = pending['parent']
    if parent_id not in record['workspaces']:
        raise integrity.InfrastructureError(f'fork parent is not registered: {name}')
    parent = root / record['workspaces'][parent_id]['path']
    stage = root / pending['staging']
    phase = pending['phase']
    context_key = f'context_{name}'
    registered = name in record['workspaces']
    manifest = integrity._load(root / 'protection')
    context_registered = context_key in manifest['roots']
    if stage_exists:
        if stage.is_symlink() or target.exists() or target.is_symlink():
            raise integrity.InfrastructureError(f'fork staging and target are ambiguous: {name}')
        if phase not in {'reservation', 'staged'}:
            raise integrity.InfrastructureError(f'fork staging contradicts its recorded phase: {name}')
        if registered or context_registered:
            raise integrity.InfrastructureError(f'unpublished fork has registered state: {name}')
        if os.stat(stage).st_dev != os.stat(target.parent).st_dev:
            raise integrity.InfrastructureError(f'fork staging filesystem changed: {name}')
        try:
            valid = workspace.validate_workspace(stage, root_mode=0o700,
                    workspace_path=target, parent_workspace=parent, case_id=record['case_id'])
        except workspace.InvalidWorkspaceError as error:
            if phase == 'reservation':
                _discard_unpublished_fork_reservation(root, record, name, pending, stage=stage)
                return
            raise integrity.InfrastructureError(f'invalid staged fork {name}: {error}') from error
        if valid is None:
            if pending['phase'] == 'reservation':
                _discard_unpublished_fork_reservation(root, record, name, pending, stage=stage)
                return
            raise integrity.InfrastructureError(f'incomplete staged fork: {name}')
        pending['phase'] = 'staged'
        _commit(root, record)
        _rename_fork_stage(stage, target)
        pending['phase'] = 'renamed-locked'
        phase = 'renamed-locked'
        _commit(root, record)
    elif target.exists() or target.is_symlink():
        if phase == 'reservation':
            raise integrity.InfrastructureError(
                f'fork target exists before staging was committed: {name}')
    else:
        if pending['phase'] == 'reservation':
            _discard_unpublished_fork_reservation(root, record, name, pending)
            return
        raise integrity.InfrastructureError(f'fork staging and target are both missing: {name}')

    if target.is_symlink():
        raise integrity.InfrastructureError(f'fork target is a symlink: {name}')
    if stage_exists and stage.exists():
        raise integrity.InfrastructureError(f'fork staging remains after rename: {name}')
    target_mode = stat.S_IMODE(target.lstat().st_mode)
    if target_mode not in {0o700, 0o3770}:
        raise integrity.InfrastructureError(f'fork target is partially published: {name}')
    allowed_registration = {
        'staged': (False, False),
        'renamed-locked': (False, None),
        'context-registered': (False, True),
        'registered-locked': (True, True),
        'registered-open': (True, True),
    }
    expected_registration = allowed_registration.get(phase)
    if (expected_registration is None or registered != expected_registration[0]
            or expected_registration[1] is not None
            and context_registered != expected_registration[1]):
        raise integrity.InfrastructureError(f'fork phase contradicts its registration state: {name}')
    if target_mode == 0o700 and phase == 'registered-open':
        raise integrity.InfrastructureError(f'published fork became inaccessible during recovery: {name}')
    if target_mode == 0o3770 and phase not in {'registered-locked', 'registered-open'}:
        raise integrity.InfrastructureError(f'fork opened before registration: {name}')
    valid = workspace.validate_workspace(target, root_mode=target_mode,
            workspace_path=target, parent_workspace=parent, case_id=record['case_id'],
            verify_candidate=(target_mode == 0o700),
            compare_starting=(target_mode == 0o700))
    if valid is None:
        raise integrity.InfrastructureError(f'fork target is incomplete: {name}')
    if target_mode == 0o3770 and not registered:
        raise integrity.InfrastructureError(f'unregistered fork is already open to the Agent: {name}')
    if registered and not context_registered:
        raise integrity.InfrastructureError(f'registered fork has no Context integrity root: {name}')
    _context_root_registered(root, name, target)
    if not registered:
        record['workspaces'][name] = {'path': f'workspaces/{name}', 'parent': parent_id}
    pending['phase'] = 'registered-locked'
    _commit(root, record)
    if target_mode == 0o700:
        workspace.publish_workspace_root(target)
    if workspace.validate_workspace(target, root_mode=0o3770,
            workspace_path=target, parent_workspace=parent, case_id=record['case_id'],
            verify_candidate=False, compare_starting=False) is None:
        raise integrity.InfrastructureError(f'published fork is incomplete: {name}')
    pending['phase'] = 'registered-open'
    _commit(root, record)
    record['pending_forks'].pop(name)
    _commit(root, record)


def recover_pending_forks(session_root):
    """Recover complete closed forks and discard only uncommitted reservations."""
    recovered = []
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root, allow_pending=True)
        forks_directory = root / 'staging/forks'
        with file_tree.open_directory(forks_directory) as descriptor:
            stage_names = set(os.listdir(descriptor))
        expected_stages = {name for name, pending in record['pending_forks'].items()
                           if (root / pending['staging']).parent == forks_directory}
        if stage_names - expected_stages:
            raise integrity.InfrastructureError('unknown fork staging directory exists')
        with file_tree.open_directory(root / 'workspaces') as descriptor:
            workspace_names = set(os.listdir(descriptor))
        expected_workspaces = set(record['workspaces']) | set(record['pending_forks'])
        if workspace_names - expected_workspaces:
            raise integrity.InfrastructureError('unregistered Workspace directory exists')
        for name in list(record['pending_forks']):
            pending = record['pending_forks'].get(name)
            if pending is None:
                continue
            stage = root / pending['staging']
            target = root / 'workspaces' / name
            stage_exists = stage.exists() or stage.is_symlink()
            target_exists = target.exists() or target.is_symlink()
            if stage_exists and target_exists:
                raise integrity.InfrastructureError(f'fork staging and target both exist: {name}')
            before = set(record['workspaces'])
            _finish_pending_fork(root, record, name, pending, stage_exists=stage_exists)
            if name in record['workspaces'] and name not in before and name not in record['pending_forks']:
                recovered.append(name)
        return recovered


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    if type(pid) is not int or pid <= 0:
        raise SessionError("process PID must be a positive integer")
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        fields = data[data.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": int(fields[19]), "pgid": int(fields[2])}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, IndexError) as error:
        raise integrity.InfrastructureError(f"cannot establish process identity: {error}") from error


def _group_alive(identity):
    # Only a separately created action group is owned by Bench. Shared parent
    # groups cannot be attributed to a single action and are never scanned/killed.
    if identity is None or identity["pgid"] != identity["pid"]:
        return False
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            member = process_identity(int(path.name))
            if member is not None and member["pgid"] == identity["pgid"]:
                return True
    return False


def _validate_source_manifest(value):
    try:
        if (not isinstance(value, dict) or set(value) != {"schema_version", "files", "directories", "content_sha256"}
                or type(value["schema_version"]) is not int or value["schema_version"] != 1
                or not isinstance(value["files"], list) or not isinstance(value["directories"], list)):
            raise ValueError("schema")
        previous = None
        for entry in value["files"]:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError("file entry")
            _inside(Path("/source"), entry["path"])
            if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
                raise ValueError("file hash")
            if previous is not None and entry["path"] <= previous:
                raise ValueError("file order")
            previous = entry["path"]
        if not value["directories"] or value["directories"][0] != "" or value["directories"] != sorted(set(value["directories"])):
            raise ValueError("directories")
        for relative in value["directories"][1:]:
            _inside(Path("/source"), relative)
        encoded = json.dumps(value["files"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if value["content_sha256"] != hashlib.sha256(encoded).hexdigest():
            raise ValueError("tree hash")
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise SessionError(f"a valid stable source manifest is required: {error}") from error


def reserve_action(session_root, event_seq, kind, *, workspace_path=None,
                   mode=None, source_manifest=None, observed_mode_changes=None,
                   compile_config=None, parent_attempt_id=None,
                   candidate_capture_skipped=None,
                   candidate_capture_skipped_versions=None):
    """Prechecks completed before this call; global lock remains held afterwards.

    A frozen manifest from S1's copy tool is mandatory. Finish callers supply a
    clean final workspace and first close exploration. This API launches nothing.
    """
    _require_execution()
    if kind not in {"evaluate", "profile", "finish"} or type(event_seq) is not int:
        raise SessionError("invalid action kind/sequence")
    if kind == "profile" and mode not in PUBLIC_MODES or kind != "profile" and mode is not None:
        raise SessionError("invalid public profiling mode")
    _validate_source_manifest(source_manifest)
    with _session_lock(session_root) as root:
        record, events = _load_locked(root)
        if kind == "finish":
            if record["status"] in {"completed", "failed"} or not record["exploration_closed"]:
                raise SessionError("finish requires closed nonterminal exploration", "session_closed")
            if record["active_action"] is not None:
                raise SessionError("active exploration must finish first")
            if record["usage"]["finish_actions"]:
                raise SessionError("finish budget already used", "budget_exceeded")
            target = _path(workspace_path)
            if target != root / "final/workspace":
                raise SessionError("finish requires the clean final workspace", "invalid_workspace")
            workspace_id = "final"
        else:
            _open_exploration(record)
            if (record["budget"]["max_hardware_actions"] != UNLIMITED_HARDWARE_ACTIONS
                    and record["usage"]["hardware_actions"] >= record["budget"]["max_hardware_actions"]):
                raise SessionError("Hardware action budget exhausted", "budget_exceeded")
            workspace_id, target = _workspace_path(root, record, workspace_path)
            if kind == "profile" and mode not in record["profiling_modes"]:
                raise SessionError("profile is not enabled in this Session")
        if event_seq != record["next_event_seq"]:
            raise SessionError("event sequence is stale")
    # Full tree/hash checks never hold the short Session record lock.
    source_skipped = []
    source_skipped_versions = {}
    workspace.source_manifest(target,
                              best_effort=candidate_capture_skipped is not None,
                              skipped_paths=source_skipped,
                              skipped_versions=source_skipped_versions)
    if (candidate_capture_skipped is not None
            and (sorted(set(source_skipped)) != sorted(set(candidate_capture_skipped))
                 or source_skipped_versions != candidate_capture_skipped_versions)):
        raise workspace.InvalidWorkspaceError('Candidate unreadable entries changed during action reservation')
    clean = integrity.check_only(root / "protection", {"fixed", "context"})
    if not clean.passed:
        raise SessionError("protected files must be repaired before reservation", "integrity_failed")
    with _session_lock(root):
        record, events = _load_locked(root)
        if kind == "finish":
            if record["status"] in {"completed", "failed"} or not record["exploration_closed"]:
                raise SessionError("finish requires closed nonterminal exploration", "session_closed")
            if record["active_action"] is not None or record["usage"]["finish_actions"]:
                raise SessionError("finish is active or already used", "budget_exceeded")
        else:
            _open_exploration(record)
            if (record["budget"]["max_hardware_actions"] != UNLIMITED_HARDWARE_ACTIONS
                    and record["usage"]["hardware_actions"] >= record["budget"]["max_hardware_actions"]):
                raise SessionError("Hardware action budget exhausted", "budget_exceeded")
        if record["next_event_seq"] != event_seq:
            raise SessionError("event sequence is stale")
        counter = "finish_actions" if kind == "finish" else "hardware_actions"
        record["usage"][counter] += 1
        if kind in {"evaluate", "profile"}:
            maximum = record["budget"]["max_hardware_actions"]
            record["status"] = ("exhausted" if maximum != UNLIMITED_HARDWARE_ACTIONS
                                and record["usage"]["hardware_actions"] >= maximum else "running")
        record["next_event_seq"] += 1
        record["active_action"] = {
            "event_seq": event_seq, "kind": kind, "mode": mode, "workspace": workspace_id,
            "content_sha256": source_manifest["content_sha256"], "started_at": _now(),
            "owner": process_identity(), "process": None,
            "process_state": "not_started",
            "observed_mode_changes": sorted(set(observed_mode_changes or [])),
            "attempt_id": 1 + sum(event["action"] in {"evaluate", "profile", "finish"} for event in events),
            "source_snapshot": str((root / ("final/workspace" if kind == "finish" else
                f"staging/action_{event_seq:06d}/workspace")).relative_to(root)),
            "parent_workspace_id": (record["workspaces"].get(workspace_id, {}).get("parent")
                                    if kind != "finish" else None),
            "parent_attempt_id": (parent_attempt_id if kind == "finish" else next((
                event.get("attempt_id") for event in reversed(events)
                if event.get("attempt_id") is not None and event.get("source_snapshot")
                and event.get("workspace") == workspace_id
            ), next(((event.get("result") or {}).get("parent_attempt_id") for event in reversed(events)
                     if event["action"] == "fork" and isinstance(event.get("result"), dict)
                     and event["result"].get("child_workspace_id") == workspace_id), None))),
            "compile_config": copy.deepcopy(compile_config),
        }
        _commit(root, record)
        return copy.deepcopy(record["usage"])


def mark_action_starting(session_root, event_seq):
    """Commit the spawn boundary before launching; S3 binds the PID afterwards."""
    _require_execution()
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root)
        active = record["active_action"]
        if active is None or active["event_seq"] != event_seq or active["process_state"] != "not_started":
            raise SessionError("no matching action awaiting spawn")
        active["process_state"] = "starting"
        _commit(root, record)


def set_action_process(session_root, event_seq, pid):
    _require_execution()
    identity = process_identity(pid)
    if identity is None:
        raise integrity.InfrastructureError("action subprocess is no longer alive")
    if identity["pgid"] != identity["pid"]:
        raise SessionError("action subprocess must own a separate process group")
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root)
        active = record["active_action"]
        if active is None or active["event_seq"] != event_seq or active["process_state"] != "starting":
            raise SessionError("no matching unbound active action")
        active["process"] = identity
        active["process_state"] = "running"
        _commit(root, record)


def complete_action(session_root, event_seq, *, exit_code, error_code=None,
                    integrity_passed=False, result=None, repaired_paths=None,
                    snapshot_id=None, action_started=True, observation=None,
                    final_result=None, record_updates=None, artifact_index=None,
                    observed_mode_changes=None, attempt_metadata=None,
                    candidate_capture_skipped=None, snapshot_error=None):
    """Atomically save the event and clear active_action, even when exhausted.

    Result validation and Snapshot creation are S3/S4 responsibilities. The
    caller supplies their evidence here; this function never executes hardware.
    """
    _require_execution()
    if (type(event_seq) is not int or type(exit_code) is not int or error_code is not None and error_code not in ERROR_CODES
            or type(integrity_passed) is not bool or type(action_started) is not bool
            or repaired_paths is not None and (not isinstance(repaired_paths, list) or any(not isinstance(p, str) for p in repaired_paths))
            or observed_mode_changes is not None and (not isinstance(observed_mode_changes, list)
                    or any(not isinstance(p, str) for p in observed_mode_changes))
            or candidate_capture_skipped is not None and (not isinstance(candidate_capture_skipped, list)
                    or any(not isinstance(p, str) or not p for p in candidate_capture_skipped)
                    or candidate_capture_skipped != sorted(set(candidate_capture_skipped)))
            or snapshot_error is not None and (not isinstance(snapshot_error, str) or not snapshot_error
                    or snapshot_id is not None or error_code is not None or exit_code != 0
                    or not integrity_passed or not action_started)):
        raise SessionError("invalid action completion evidence")
    with _session_lock(session_root) as root:
        record, events = _load_locked(root)
        active = record["active_action"]
        existing = next((e for e in events if e["seq"] == event_seq), None)
        if existing is not None:
            return copy.deepcopy(existing)
        if active is None:
            raise SessionError("no matching active action")
        if active["event_seq"] != event_seq:
            raise SessionError("active action sequence mismatch")
        if (candidate_capture_skipped is not None and active["kind"] not in {"evaluate", "profile"}
                or snapshot_error is not None and active["kind"] not in {"evaluate", "profile"}):
            raise SessionError("Candidate archive evidence is valid only for Evaluate/Profile")
        event = {
            "run_id": record["session_id"], "seq": event_seq, "time": _now(),
            "action": active["kind"], "mode": active["mode"],
            "started_at": active['started_at'], "process": active['process'],
            "artifact_index": artifact_index,
            "workspace": active["workspace"], "content_sha256": active["content_sha256"],
            "exit_code": exit_code, "error_code": error_code, "hardware_action": True,
            "action_started": action_started, "integrity_passed": integrity_passed,
            "repaired_paths": repaired_paths or [], "result": result, "snapshot_id": snapshot_id,
            "observed_mode_changes": sorted(set(observed_mode_changes or [])
                                             | set(active.get("observed_mode_changes", []))),
            "hardware_actions_used": record["usage"]["hardware_actions"],
        }
        if candidate_capture_skipped:
            event["candidate_capture_skipped"] = list(candidate_capture_skipped)
        if snapshot_error is not None:
            event["snapshot_error"] = snapshot_error
        if "attempt_id" in active:
            event.update(attempt_id=active["attempt_id"],
                         source_snapshot=active["source_snapshot"],
                         parent_workspace_id=active.get("parent_workspace_id"),
                         parent_attempt_id=active.get("parent_attempt_id"),
                         compile_config=copy.deepcopy(active.get("compile_config")),
                         compile_status="unavailable", correctness_status="unavailable",
                         failure_stage="unavailable" if error_code else None,
                         bench_status="not_run",
                         diagnostic_path=artifact_index, latency_samples=[], latency_samples_by_fixture={},
                         latency_us=None, speedup=None, measurement_config={}, device_latency_samples_us={},
                         bench_sampled_device_time_us=None, runtime_dispatch_entries=None,
                         bench_sampled_rounds=None,
                         profile_artifacts_path=artifact_index if active["kind"] == "profile" else None)
            if attempt_metadata:
                event.update(copy.deepcopy(attempt_metadata))
        if observation is not None:
            event["observation"] = copy.deepcopy(observation)
        event["state"] = "completed"
        record["active_action"] = None
        if active["kind"] == "finish":
            record["status"] = "completed" if exit_code == 0 and error_code is None and integrity_passed else "failed"
            record["completed_at"] = _now()
        if final_result is not None:
            if active['kind'] != 'finish' or final_result['status'] != record['status']:
                raise SessionError('final result must match finish completion')
            final_result['completed_at'] = record['completed_at']
            final_result['elapsed_seconds'] = elapsed_seconds(record['created_at'], record['completed_at'])
            record_updates = {**(record_updates or {}), 'final_result': _bytes(final_result)}
        events.append(event)
        if record_updates:
            _commit(root, record, events, record_updates=record_updates)
        else:
            _commit(root, record, events)
        return copy.deepcopy(event)


def close_exploration(session_root, reason):
    if not isinstance(reason, str) or not reason:
        raise SessionError("a close reason is required")
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root)
        if not record["exploration_closed"]:
            record["exploration_closed"] = True
            record["close_reason"] = reason
            _commit(root, record)


def reconcile_active_action(session_root):
    """Launcher recovery: record a dead action once; refuse live/unknown workers.

    S3/S5 owns terminating confirmed child process groups. S2 never kills an
    unrelated process. An ambiguous live identity makes the Session terminal,
    keeps active_action as evidence, and blocks final measurement.
    """
    _require_execution()
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root)
        active = record["active_action"]
        if active is None:
            return None
        owner = active["owner"]
        process = active["process"]
        live_owner = owner is not None and process_identity(owner["pid"]) == owner
        live_child = process is not None and process_identity(process["pid"]) == process
        live_group = _group_alive(process)
        if live_owner:
            raise SessionError("the accepted action is still running")
        ambiguous = active["process_state"] == "starting"
        if live_child or live_group or ambiguous:
            record["exploration_closed"] = True
            record["close_reason"] = "unresolved_active_process"
            record["status"] = "failed"
            record["completed_at"] = _now()
            _commit(root, record)
            raise integrity.InfrastructureError("active process still alive or cannot be attributed; final execution blocked")
        seq = active["event_seq"]
        record["exploration_closed"] = True
        record["close_reason"] = record["close_reason"] or "interrupted"
        _commit(root, record)
    return complete_action(root, seq, exit_code=130, error_code="interrupted",
                           action_started=active["process_state"] != "not_started")


def record_nonhardware_event(session_root, action, *, workspace_id=None, mode=None,
                             error_code=None, exit_code=0, repaired_paths=None, result=None,
                             observed_mode_changes=None):
    """S3 fork/rejection event allocation and commit; never consumes an action."""
    _require_execution()
    if action not in {"evaluate", "profile", "fork", "invalid_request"}:
        action = "invalid_request"
    with _session_lock(session_root) as root:
        record, events = _load_locked(root)
        seq = record["next_event_seq"]
        record["next_event_seq"] += 1
        event = {
            "run_id": record["session_id"], "seq": seq, "time": _now(), "action": action, "mode": mode,
            "workspace": workspace_id, "content_sha256": None, "exit_code": exit_code,
            "error_code": error_code, "hardware_action": False, "action_started": False,
            "integrity_passed": error_code is None, "repaired_paths": repaired_paths or [],
            "observed_mode_changes": sorted(set(observed_mode_changes or [])),
            "result": result, "snapshot_id": None,
            "hardware_actions_used": record["usage"]["hardware_actions"],
            "state": "rejected" if error_code else "completed",
        }
        if action in {"evaluate", "profile"}:
            event.update(
                attempt_id=1 + sum(item["action"] in {"evaluate", "profile", "finish"} for item in events),
                source_snapshot=None, parent_workspace_id=None, parent_attempt_id=None, compile_config=None,
                compile_status="not_run", correctness_status="not_run",
                failure_stage="request" if error_code else None, diagnostic_path=None,
                bench_status="not_run", latency_samples=[], latency_samples_by_fixture={},
                latency_us=None, speedup=None, measurement_config={},
                device_latency_samples_us={}, bench_sampled_device_time_us=None,
                runtime_dispatch_entries=None, bench_sampled_rounds=None,
                profile_artifacts_path=None)
        elif action == "fork" and workspace_id in record["workspaces"]:
            info = record["workspaces"][workspace_id]
            event.update(parent_workspace_id=info["parent"],
                         parent_attempt_id=(event["result"].get("parent_attempt_id")
                                            if isinstance(event["result"], dict) else None),
                         source_snapshot=(event["result"].get("source_snapshot")
                                          if isinstance(event["result"], dict) else None))
        events.append(event)
        _commit(root, record, events)
        return copy.deepcopy(event)


def fail_session(session_root, reason):
    """Close a readable Session after an infrastructure failure, keeping evidence."""
    _require_execution()
    with _session_lock(session_root) as root:
        record, _ = _load_locked(root)
        record.update(status="failed", exploration_closed=True, close_reason=reason, completed_at=_now())
        _commit(root, record)
