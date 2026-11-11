"""Fail-closed checks for the one PTOBench execution environment.

The lock is deliberately about installed tools and runtime binaries, not the
PyPTO-Lib source tree.  Cases and Agent candidates are allowed to change that
tree; changing the compiler/runtime stack requires an explicit lock refresh.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


LOCK_PATH = Path(__file__).resolve().parents[1] / "environments" / "LOCK.json"


class EnvironmentMismatch(RuntimeError):
    """The selected Case is not using the certified toolchain."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _load() -> dict:
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EnvironmentMismatch(f"PTOBench environment lock is unavailable: {error}") from error


def _resolved(value: str | os.PathLike[str]) -> str:
    return str(Path(value).expanduser().resolve())


def _package_files_sha256(distribution) -> str:
    digest = hashlib.sha256()
    for relative in sorted(str(path) for path in (distribution.files or ())):
        path = distribution.locate_file(relative)
        if not path.is_file():
            raise EnvironmentMismatch(f"installed package file is missing: {path}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_diff_sha256(root: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=root,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _package_check(lock: dict, name: str) -> None:
    expected = lock["packages"][name]
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise EnvironmentMismatch(f"required package is missing: {name}") from error
    if distribution.version != expected["version"]:
        raise EnvironmentMismatch(
            f"{name} version mismatch: {distribution.version} != {expected['version']}"
        )
    direct_url = distribution.read_text("direct_url.json")
    actual_url = None if not direct_url else json.loads(direct_url).get("url")
    if actual_url != expected.get("direct_url"):
        raise EnvironmentMismatch(f"{name} installation source mismatch")
    if _package_files_sha256(distribution) != expected["files_sha256"]:
        raise EnvironmentMismatch(f"{name} installed files do not match the locked build")
    source = expected.get("source")
    if source is not None:
        root = Path(source["root"])
        if not root.is_dir():
            raise EnvironmentMismatch(f"locked {name} source is unavailable: {root}")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
        if head != source["git_head"] or _git_diff_sha256(root) != source["git_diff_sha256"]:
            raise EnvironmentMismatch(f"{name} source revision or working tree changed")


def validate(profile: dict, environment: dict[str, str] | None = None) -> None:
    """Validate the selected profile and installed runtime before execution."""
    lock = _load()
    if lock.get("schema_version") != 1:
        raise EnvironmentMismatch("unsupported PTOBench environment lock")
    expected_python = _resolved(lock["python"]["executable"])
    if _resolved(sys.executable) != expected_python:
        raise EnvironmentMismatch(f"Python mismatch: {sys.executable} != {expected_python}")
    actual_python_hash = _sha256(Path(sys.executable).resolve())
    if actual_python_hash != lock["python"]["sha256"]:
        raise EnvironmentMismatch("Python executable does not match the locked build")

    for name in ("pypto", "simpler", "torch", "PyYAML"):
        _package_check(lock, name)

    env = os.environ if environment is None else environment
    expected_ptoas = _resolved(lock["toolchain"]["ptoas"])
    actual_ptoas = _resolved(env.get("PTOAS_ROOT", "") + "/bin/ptoas") if env.get("PTOAS_ROOT") else ""
    if actual_ptoas != expected_ptoas or _sha256(Path(actual_ptoas)) != lock["toolchain"]["ptoas_sha256"]:
        raise EnvironmentMismatch("PTOAS compiler does not match the locked build")
    expected_cann = _resolved(lock["toolchain"]["cann_root"])
    actual_cann = _resolved(env.get("ASCEND_HOME_PATH", "")) if env.get("ASCEND_HOME_PATH") else ""
    if actual_cann != expected_cann:
        raise EnvironmentMismatch("CANN root does not match the locked build")

    runtime = lock["runtime"]
    runtime_root = Path(runtime["source_root"])
    if not runtime_root.is_dir():
        raise EnvironmentMismatch(f"locked runtime source is unavailable: {runtime_root}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=runtime_root,
                          check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    if head != runtime["git_head"] or _git_diff_sha256(runtime_root) != runtime["git_diff_sha256"]:
        raise EnvironmentMismatch("runtime source revision or trace patch changed")
    if _sha256(runtime_root / "pto_isa.pin") != runtime["pto_isa_pin_sha256"]:
        raise EnvironmentMismatch("PTO ISA pin changed")
    assets = Path(runtime["installed_assets"])
    if _tree_sha256(assets / "build" / "lib") != runtime["installed_build_lib_sha256"]:
        raise EnvironmentMismatch("installed simpler runtime binaries changed")

    profile_name = profile.get("name")
    expected_profile = lock["profiles"].get(profile_name)
    profile_path = Path(__file__).resolve().parents[1] / "environments" / f"{profile_name}.json"
    if expected_profile is None or _sha256(profile_path) != expected_profile:
        raise EnvironmentMismatch(f"Case environment profile changed: {profile_name}")


def lock_path() -> Path:
    return LOCK_PATH
