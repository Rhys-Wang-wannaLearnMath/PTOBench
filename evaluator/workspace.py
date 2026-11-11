"""Candidate workspace creation and workspace metadata helpers."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path

import common
import file_tree
import profile_modes
import tuning


def _case_files(case_dir: Path, case: dict) -> dict[str, Path]:
    return {
        "case.json": case_dir / "case.json",
        "config.py": case_dir / case["config"],
        "comparator.json": case_dir / case["comparator"],
    }


def _readonly(path: Path) -> None:
    entries = file_tree.scan_tree(path, reject_special=True)
    for relative, entry in entries.items():
        if entry["kind"] == "file":
            with file_tree.open_file(path / relative) as descriptor:
                os.fchmod(descriptor, 0o444)


def _reference_summary(sealed: Path) -> dict:
    certification = common.read_json(sealed / "certification.json")
    result = {"schema_version": 1, "status": certification.get("status", "pending")}
    performance = sealed / "ref_performance.json"
    if performance.exists():
        data = common.read_json(performance)
        result["reference_performance"] = {
            name: values.get("median_effective_us")
            for name, values in data.get("fixtures", {}).items()
            if "median_effective_us" in values
        }
        result["compiler_tuning"] = data.get("compiler_tuning", common.TUNING_DEFAULTS)
    return result


def _environment_summary(case: dict) -> dict:
    profile = common.read_json(
        common.BENCHMARK_ROOT / "environments" / f"{case['environment']}.json"
    )
    return {"profile": profile["name"], "platform": profile["platform"],
            "device": profile.get("device", 0), "runtime": profile.get("runtime", {})}


def _readme(case: dict, enabled_profiles: dict[str, dict[str, str]], *,
            profile_strategy=None, capabilities=None) -> str:
    profile_commands = ""
    diagnostics = ""
    output_description = (
        "`.benchmark_outputs/` contains read-only result and diagnostic copies produced by the CLI. "
        "The CLI refreshes these files after each action."
    )
    # The guide needs policy permissions, not a second environment lookup.
    capabilities = capabilities or tuning.agent_capabilities(case, {}, {})
    controls = capabilities['options']
    tuning_rows = '\n'.join(
        f"| `{name}` | {'compile optimization' if spec['stage'] == 'compile' else 'runtime scheduling'} | {spec['type']}; {spec['unit']} | `{json.dumps(spec['default'])}` | `{json.dumps(spec['values'])}` | {spec['description']} {spec['restrictions']} |"
        for name, spec in controls.items() if spec['editable'])
    verification = capabilities['default_configuration_validation']['status']
    if enabled_profiles:
        profile_commands = "\n".join(
            f"| `ptobench profile{'' if name == 'combined' else ' ' + name} .` | {values['description']} | "
            f"`.benchmark_outputs/{(profile_modes.COMBINED_PUBLIC_TO_INTERNAL if name == 'combined' else profile_modes.PUBLIC_TO_INTERNAL)[name]}/` |"
            for name, values in enabled_profiles.items()
        ) + "\n"
        diagnostics = " Profile does not check output correctness. Use `ptobench evaluate` to check correctness and clean latency."
        output_description = (
            "`.benchmark_outputs/` contains read-only result and diagnostic copies produced by the CLI. "
            "`evaluate` writes clean measurement results; `profile` writes diagnostic outputs. "
            "The CLI refreshes these files after each action."
        )
    return f"""# Benchmark workspace

Optimize Case `{case['case_id']}`: minimize latency while preserving correctness.
Candidate entry: `candidate/{case['module']}::{case['entry']}`. Preserve the inputs, outputs,
shapes, dtypes and comparison rules in the public contract below.
Edit the Candidate Python files and local helpers under `candidate/`, plus the
supported options in `candidate/tuning.json`. The task contract is in
`.benchmark_context/`. Use the PTOBench
CLI to evaluate changes and collect diagnostics. PATH and `PTOBENCH_SESSION_ROOT`
are already set by the launcher.

## Files

The following files are read-only and relative to `.benchmark_context/`:

| File | Purpose |
| --- | --- |
| `case.json`, `config.py`, `comparator.json` | Case description, fixed inputs/outputs/configuration and correctness rules. |
| `reference.json` | Certified baseline latency in microseconds; no initial evaluation is required. |
| `environment.json` | Target platform and device. |
| `tuning_capabilities.json` | Per-Case tuning permissions and inherited defaults. |
| `workspace.json` | Current workspace path, entry and parent workspace. |
| `starting_point/` | Source copy from when this workspace was created. |

`candidate/tuning.json` is editable workspace configuration:

| Parameter | Stage | Type; unit | Default | Allowed values | Effect and applicability |
| --- | --- | --- | --- | --- | --- |
{tuning_rows}

The listed options are the values available in `tuning.json`; the set and defaults
vary by Case. Reference and baseline measurements use the registered defaults,
including `PYPTO + false`. Default Candidate correctness and clean baseline
verification: **{verification}**. After changing Candidate source or tuning,
run `ptobench evaluate .` to verify correctness and measure clean latency.
`enable_dep_gen` is a separate diagnostic option. If `tuning.json` is absent,
the defaults apply.

{output_description}

## Commands

| Command | Purpose | Output |
| --- | --- | --- |
| `ptobench evaluate .` | Check correctness and measure clean latency/speedup against the baseline. | `.benchmark_outputs/clean/` |
{profile_commands}| `ptobench fork .` | Copy implementation into a new registered workspace; returns its absolute path. | New workspace |

Replace `.` with another registered workspace path, or omit it to use the current
directory. Use a workspace root, not a subdirectory. After fork, edit and evaluate the returned workspace;
its changes are independent of the parent. `ptobench --help` shows available commands.
Each accepted evaluation or enabled diagnostic operation uses one exploration-budget action, including
unchanged-source runs, compilation failures and interruptions. This counts the
Agent's experiment attempt, not only NPU executions; observable runtime dispatches
and measured device time are recorded separately. Rejected requests and fork cost
nothing. All workspaces and Agent stages share the same Bench Session budget.

## Read feedback

Each command ends with JSON: `status` is PASS, FAIL or REJECTED; `error_code` and
`message` explain failures. `latency_us` is Candidate median latency in microseconds
(lower is better); `remaining_hardware_actions` reports the remaining budget.
Fork returns `workspace`, the new registered root.{diagnostics}
In the output directory, `latest_result.json` contains correctness, raw samples,
speedup (higher is better) and available artifact/build paths.
`latest_session_result.json` contains the command status, latency and budget feedback.
Use the returned artifact/build paths to locate detailed compiler and diagnostic files.
"""


def _assert_destination(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"workspace is not empty: {destination}")


def _ensure_tuning(destination: Path, case: dict | None = None) -> None:
    """Create the editable tuning file for newly initialized workspaces."""
    path = destination / "candidate" / "tuning.json"
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise InvalidWorkspaceError(f"tuning.json is not a regular file: {path}")
        return
    data = (json.dumps(common.tuning_defaults(case) if case else common.TUNING_DEFAULTS,
                       indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    file_tree.atomic_bytes(path, data, mode=0o644)


CANDIDATE_EXCLUDES = frozenset({
    ".benchmark_context", ".benchmark_outputs", ".git", ".gitkeep", ".pytest_cache", "__pycache__",
})


class InvalidWorkspaceError(ValueError):
    error_code = "invalid_workspace"


def _source_entries(source: Path, *, excluded_paths=frozenset()) -> dict[str, dict]:
    entries = file_tree.scan_tree(source, excluded_names=CANDIDATE_EXCLUDES,
                                  reject_special=True, excluded_paths=excluded_paths)
    if entries[""]["kind"] != "directory":
        raise InvalidWorkspaceError(f"Candidate source is not a directory: {source}")
    return entries


def candidate_root(source: Path) -> Path:
    """Return the Candidate payload root for a workspace or legacy seed."""
    source = file_tree.absolute_path(source)
    candidate = source / "candidate"
    if candidate.is_dir() and not candidate.is_symlink():
        return candidate
    return source


def _candidate_payload_entries(source: Path, *, excluded_paths=frozenset()) -> dict[str, dict]:
    entries = _source_entries(source, excluded_paths=excluded_paths)
    for relative, entry in entries.items():
        if entry['kind'] == 'file' and relative != 'tuning.json' and Path(relative).suffix != '.py':
            raise InvalidWorkspaceError(
                f"workspace file is outside the Candidate and tuning.json: {relative}"
            )
    return entries


def _candidate_entries(source: Path, *, excluded_paths=frozenset()) -> dict[str, dict]:
    return _candidate_payload_entries(candidate_root(source), excluded_paths=excluded_paths)


def _source_manifest(entries: dict[str, dict]) -> dict:
    files = [
        {"path": path, "sha256": entry["sha256"]}
        for path, entry in sorted(entries.items()) if entry["kind"] == "file"
    ]
    serialized = json.dumps(files, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
    return {"schema_version": 1, "files": files,
            "directories": sorted(path for path, entry in entries.items()
                                  if entry["kind"] == "directory"),
            "content_sha256": hashlib.sha256(serialized).hexdigest()}


def source_manifest(source: Path) -> dict:
    """Canonical source inventory/hash shared by staging and Snapshots."""
    try:
        return _source_manifest(_candidate_entries(file_tree.absolute_path(source)))
    except (OSError, UnicodeError, file_tree.UnsafeTreeError, file_tree.TreeChangedError) as error:
        raise InvalidWorkspaceError(str(error)) from error


def _copy_source_file(source: Path, target: Path, entry: dict) -> None:
    with file_tree.open_file(source) as source_fd, \
            file_tree.open_parent(target) as (parent, name):
        before = os.fstat(source_fd)
        target_fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            mode=0o600, dir_fd=parent)
        try:
            file_tree.copy_descriptor(source_fd, target_fd)
            if (file_tree.file_version(before) != file_tree.file_version(os.fstat(source_fd))
                    or file_tree.descriptor_sha256(target_fd) != entry["sha256"]):
                raise file_tree.TreeChangedError(f"Candidate changed during copy: {source}")
            os.fchmod(target_fd, entry["mode"])
        finally:
            os.close(target_fd)


def _copy_source_tree(source: Path, staging: Path, entries: dict[str, dict]) -> None:
    directories = [path for path, entry in entries.items() if entry["kind"] == "directory"]
    for relative in sorted(directories, key=lambda path: (path.count("/"), path)):
        with file_tree.open_directory(staging / relative, create=True):
            pass
    for relative, entry in sorted(entries.items()):
        if entry["kind"] == "file":
            _copy_source_file(source / relative, staging / relative, entry)
    for relative in sorted(directories, key=lambda path: path.count("/"), reverse=True):
        with file_tree.open_directory(staging / relative) as descriptor:
            os.fchmod(descriptor, entries[relative]["mode"])


def _copy_tree(source: Path, destination: Path, *, max_retries: int, candidate: bool) -> dict:
    """Copy a stable tree without publishing partial data."""
    source, destination = map(file_tree.absolute_path, (source, destination))
    if type(max_retries) is not int or not 0 <= max_retries <= 3:
        raise InvalidWorkspaceError("max_retries must be an integer from 0 to 3")
    if destination == source or source.is_relative_to(destination):
        raise InvalidWorkspaceError("Candidate source and destination overlap")
    if destination.is_relative_to(source):
        first = destination.relative_to(source).parts[0]
        if first not in CANDIDATE_EXCLUDES:
            raise InvalidWorkspaceError("destination is inside the Candidate source")
    scan_source = candidate_root(source) if candidate else source
    scan = _candidate_entries if candidate else _source_entries
    try:
        with file_tree.open_directory(destination.parent, create=True) as parent:
            for attempt in range(max_retries + 1):
                staging_name = f".{destination.name}.copy_{uuid.uuid4().hex}"
                os.mkdir(staging_name, mode=0o700, dir_fd=parent)
                staging = destination.parent / staging_name
                try:
                    entries = scan(scan_source)
                    _copy_source_tree(scan_source, staging, entries)
                    if entries != scan(scan_source) or entries != scan(staging):
                        raise file_tree.TreeChangedError("Candidate source changed during copy")
                    try:
                        target = os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        if not stat.S_ISDIR(target.st_mode):
                            raise InvalidWorkspaceError(f"target is not a real directory: {destination}")
                        with file_tree.open_directory(destination) as descriptor:
                            if os.listdir(descriptor):
                                raise InvalidWorkspaceError(f"workspace is not empty: {destination}")
                    manifest = _source_manifest(entries)
                    os.rename(staging_name, destination.name, src_dir_fd=parent, dst_dir_fd=parent)
                    return manifest
                except (file_tree.TreeChangedError, FileNotFoundError) as error:
                    if attempt == max_retries:
                        raise InvalidWorkspaceError(f"Candidate could not be copied stably: {error}") from error
                finally:
                    file_tree.remove_at(parent, staging_name)
    except (OSError, UnicodeError, file_tree.UnsafeTreeError) as error:
        raise InvalidWorkspaceError(str(error)) from error


def copy_candidate_tree(source: Path, destination: Path, *, max_retries: int = 3) -> dict:
    """Copy Candidate Python sources and tuning.json, excluding task/results."""
    return _copy_tree(source, destination, max_retries=max_retries, candidate=True)


def copy_tree(source: Path, destination: Path, *, max_retries: int = 3) -> dict:
    """Copy a general artifact tree using the same stable publication checks."""
    return _copy_tree(source, destination, max_retries=max_retries, candidate=False)


def write_context(case_dir: Path, sealed: Path, case: dict, destination: Path,
                  parent_workspace: Path | None = None,
                  workspace_path: Path | None = None, *, enabled_profiles=None,
                  profile_strategy=None) -> None:
    metadata = destination / ".benchmark_context"
    with file_tree.open_directory(metadata, create=True):
        pass
    copy_candidate_tree(destination, metadata / "starting_point" / "candidate")
    available = profile_modes.public_enabled_modes(profile_strategy)
    enabled_profiles = (available if enabled_profiles is None else
                        {name: available[name] for name in enabled_profiles if name in available})
    for name, source in _case_files(case_dir, case).items():
        with file_tree.open_file(source) as descriptor:
            with os.fdopen(os.dup(descriptor), "rb") as reader:
                file_tree.atomic_bytes(metadata / name, reader.read())
    binding = {
        "schema_version": 1, "case_id": case["case_id"],
        "candidate_module": case["module"], "entry": case["entry"],
        "workspace": str((workspace_path or destination).resolve()),
        "parent_workspace": str(parent_workspace.resolve()) if parent_workspace else None,
    }
    def write_json(name, value):
        data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        file_tree.atomic_bytes(metadata / name, data)

    write_json("workspace.json", binding)
    environment_summary = _environment_summary(case)
    performance_path = sealed / 'ref_performance.json'
    performance = common.read_json(performance_path) if performance_path.is_file() else {}
    current_identity = None
    if performance.get('tuning_identity'):
        from verify_tuning import identity
        current_identity = identity(case_dir, case)
    capabilities = tuning.agent_capabilities(case, environment_summary, performance, current_identity)
    reference = _reference_summary(sealed)
    reference['default_configuration_validation'] = capabilities['default_configuration_validation']
    write_json("reference.json", reference)
    write_json("environment.json", environment_summary)
    write_json("tuning_capabilities.json", capabilities)
    file_tree.atomic_bytes(metadata / "README.md", _readme(
        case, enabled_profiles, profile_strategy=profile_strategy,
        capabilities=capabilities).encode("utf-8"))
    _readonly(metadata)


def init_workspace(case_dir: Path, sealed: Path, case: dict, destination: Path, **context_options) -> None:
    destination = file_tree.absolute_path(destination)
    if destination.is_symlink():
        raise InvalidWorkspaceError(f"workspace cannot be a symlink: {destination}")
    _assert_destination(destination)
    copy_candidate_tree(case_dir / case["workspace"], destination / "candidate")
    _ensure_tuning(destination, case)
    write_context(case_dir, sealed, case, destination, **context_options)


def fork_workspace(parent: Path, destination: Path) -> dict[str, str | None]:
    parent = file_tree.absolute_path(parent)
    destination_input = Path(destination)
    if destination_input.is_symlink():
        raise SystemExit(f"child workspace cannot be a symlink: {destination_input}")
    destination = file_tree.absolute_path(destination_input)
    if not parent.is_dir():
        raise SystemExit(f"not a benchmark workspace: {parent}")
    if not (parent / ".benchmark_context" / "workspace.json").is_file():
        raise SystemExit(f"not a benchmark workspace: {parent}")
    if destination == parent or destination.is_relative_to(parent):
        raise SystemExit("child workspace must not be the parent or inside the parent")
    _assert_destination(destination)
    try:
        with file_tree.open_file(parent / ".benchmark_context" / "workspace.json") as descriptor:
            with os.fdopen(os.dup(descriptor), "rb") as reader:
                binding = json.load(reader)
    except (OSError, file_tree.UnsafeTreeError) as error:
        raise InvalidWorkspaceError(str(error)) from error
    case_dir, sealed, case = common.resolve_case(binding["case_id"])

    with file_tree.open_directory(destination.parent, create=True) as descriptor:
        temporary_name = f".{destination.name}.fork_{uuid.uuid4().hex}"
        os.mkdir(temporary_name, mode=0o700, dir_fd=descriptor)
    temporary = destination.parent / temporary_name
    try:
        copy_candidate_tree(parent, temporary / "candidate")
        _ensure_tuning(temporary, case)
        write_context(case_dir, sealed, case, temporary, parent_workspace=parent,
                       workspace_path=destination)
        with file_tree.open_directory(destination.parent) as descriptor:
            os.rename(temporary_name, destination.name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
    except BaseException:
        with file_tree.open_directory(destination.parent) as descriptor:
            file_tree.remove_at(descriptor, temporary_name)
        raise
    return {
        "case_id": case["case_id"],
        "parent_workspace": str(parent),
        "workspace": str(destination),
    }


def resolve_workspace(value: str | Path) -> tuple[Path, str]:
    workspace = Path(value).resolve()
    binding = common.read_json(workspace / ".benchmark_context" / "workspace.json")
    return workspace, binding["case_id"]


def artifact_dir(value: str | Path, profile_mode: str) -> Path:
    """Return the evaluator-owned artifact directory for one Workspace mode."""
    return Path(value).resolve() / ".benchmark_outputs" / profile_mode
