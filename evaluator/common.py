"""Small shared helpers for v0.2 Case evaluators."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def _nested_repo_root() -> Path | None:
    if BENCHMARK_ROOT.parent.name == "benchmarks":
        root = BENCHMARK_ROOT.parents[1]
        if (root / "golden").is_dir():
            return root
    return None


def _repo_root() -> Path:
    """Return the PyPTO-Lib checkout used by a benchmark.

    In the monorepo the benchmark lives at ``pypto-lib/benchmarks/<name>``.
    A standalone PTOBench checkout has no fixed relationship to that source
    tree, so callers can point at the PyPTO-Lib checkout explicitly.
    """
    override = os.environ.get("PTOBENCH_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    nested_root = _nested_repo_root()
    if nested_root is not None:
        return nested_root
    # A standalone checkout is its own default root.  Users evaluating Cases
    # against a separate PyPTO-Lib source tree should set PTOBENCH_REPO_ROOT.
    return BENCHMARK_ROOT


REPO_ROOT = _repo_root()
# Session preflight imports official Fixture metadata before environment
# bootstrap. Make the selected checkout's golden helpers available even when
# the caller starts outside that checkout and has no inherited PYTHONPATH.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def profile_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a profile path relative to the external source checkout."""
    formatted = str(value).format(**profile_values())
    path = Path(formatted).expanduser()
    # A venv interpreter is usually a symlink. Resolving it to the base Python
    # loses the venv's installed dependencies when bootstrap re-execs it.
    return path if path.is_absolute() else REPO_ROOT / path


def profile_values() -> dict[str, str]:
    """Values available to environment profiles and setup scripts."""
    nested_defaults = _nested_repo_root() is not None
    return {
        "repo_root": str(REPO_ROOT),
        "benchmark_root": str(BENCHMARK_ROOT),
        "ptoas_root": os.environ.get(
            "PTOBENCH_PTOAS_ROOT",
            "/data1/home/ziruiwang/ptoas-v0.60" if nested_defaults else "",
        ),
        "cann_root": os.environ.get(
            "PTOBENCH_CANN_ROOT",
            "/usr/local/Ascend/cann-9.0.0/cann" if nested_defaults else "",
        ),
        "miniforge_root": os.environ.get(
            "PTOBENCH_MINIFORGE_ROOT",
            "/data1/home/ziruiwang/miniforge3" if nested_defaults else "",
        ),
    }


def sealed_root() -> Path:
    """Return the protected Case-asset tree, optionally mounted externally."""
    override = os.environ.get("PTOBENCH_SEALED_ROOT")
    return (Path(override).expanduser().resolve() if override
            else BENCHMARK_ROOT / "sealed_assets")


def await_action_start() -> None:
    """Internal S3 launch gate; maintenance calls without it proceed normally."""
    value = os.environ.pop("PTOBENCH_ACTION_START_FD", None)
    if value is None:
        return
    descriptor = int(value)
    if descriptor <= 2:
        raise RuntimeError("invalid action start descriptor")
    try:
        if os.read(descriptor, 1) != b"G":
            raise RuntimeError("action owner did not authorize Evaluator start")
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparator_path(case_dir: Path, case: dict[str, Any]) -> Path:
    return case_dir / case["comparator"]


def comparator_sha256(case_dir: Path, case: dict[str, Any]) -> str:
    return sha256_file(comparator_path(case_dir, case))


def resolve_case(value: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    requested = Path(value)
    if requested.is_dir():
        case_dir = requested.resolve()
    else:
        family, name = str(value).split(".", 1)
        case_dir = (BENCHMARK_ROOT / "cases" / family / name).resolve()
    case = read_json(case_dir / "case.json")
    relative_case = case_dir.relative_to(BENCHMARK_ROOT / "cases")
    sealed = (sealed_root() / relative_case).resolve()
    return case_dir, sealed, case


def load_module(path: Path, name: str, search_paths: tuple[Path, ...] = ()):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    added_paths = [*(str(value) for value in search_paths), str(path.parent)]
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.path[:0] = added_paths
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        del sys.path[:len(added_paths)]
    return module


def load_official_fixture_specs(
    path: Path, name: str, seed: int, *, config_alias=None,
):
    """Run a byte-exact historical TensorSpec builder under the current API."""
    import golden
    import torch

    previous_config = sys.modules.get("config")
    original_tensor_spec = golden.TensorSpec

    def current_tensor_spec(*args, is_output=False, **kwargs):
        del is_output
        return original_tensor_spec(*args, **kwargs)

    # Keep both dependency isolation and the official search path alive through
    # construction: historical builders may import helpers lazily.
    with _isolated_module_scope((path.parent,)):
        previous_path = sys.path[:]
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.path.insert(0, str(path.parent))
        sys.dont_write_bytecode = True
        if config_alias is not None:
            sys.modules["config"] = config_alias
        golden.TensorSpec = current_tensor_spec
        try:
            official = load_module(path, name)
            torch.manual_seed(seed)
            return official.build_tensor_specs()
        finally:
            golden.TensorSpec = original_tensor_spec
            sys.path[:] = previous_path
            sys.dont_write_bytecode = previous_dont_write_bytecode
            if config_alias is not None:
                if previous_config is None:
                    sys.modules.pop("config", None)
                else:
                    sys.modules["config"] = previous_config


def _local_module_names(directory: Path) -> set[str]:
    names = {path.stem for path in directory.glob("*.py") if path.name != "__init__.py"}
    names.update(path.name for path in directory.iterdir() if path.is_dir())
    return names


@contextlib.contextmanager
def _isolated_module_scope(directories: tuple[Path, ...]) -> Iterator[None]:
    """Temporarily isolate same-named imports from the given local trees."""
    local_names = set()
    for directory in directories:
        if directory.is_dir():
            local_names.update(_local_module_names(directory))
    matches = lambda module_name: any(
        module_name == local or module_name.startswith(f"{local}.")
        for local in local_names
    )
    previous = {
        module_name: module
        for module_name, module in tuple(sys.modules.items())
        if matches(module_name)
    }
    for module_name in previous:
        del sys.modules[module_name]
    try:
        yield
    finally:
        for module_name in tuple(sys.modules):
            if matches(module_name):
                del sys.modules[module_name]
        sys.modules.update(previous)


def load_isolated_module(
    path: Path, name: str, search_paths: tuple[Path, ...] = (),
):
    """Load a local module tree without reusing or replacing same-named imports."""
    with _isolated_module_scope((path.parent, *search_paths)):
        return load_module(path, name, search_paths)


def load_case_evaluator(sealed: Path):
    return load_module(sealed / "evaluate.py", f"case_evaluator_{sealed.name.replace('.', '_')}")


def load_candidate(case_dir: Path, case: dict[str, Any], workspace: Path, name: str):
    """Load a Candidate entry with its workspace-local dependency closure."""
    return load_isolated_module(
        workspace / case["module"], name, (case_dir / "public", workspace),
    )


def load_ref(case_dir: Path, sealed: Path, name: str):
    """Load the frozen NPU Ref with only its own implementation dependencies."""
    ref_dir = sealed / "ref"
    return load_isolated_module(
        ref_dir / "ref_npu.py", name, (case_dir / "public", ref_dir),
    )


def load_oracle(case_dir: Path, sealed: Path, name: str):
    """Load an Oracle with its own CPU dependency closure."""
    oracle_dir = sealed / "oracle"
    return load_isolated_module(
        oracle_dir / "ref_cpu.py", name, (case_dir / "public", oracle_dir),
    )


def _environment_profile(case: dict[str, Any]) -> dict[str, Any]:
    return read_json(BENCHMARK_ROOT / "environments" / f"{case['environment']}.json")


def bootstrap_environment(case: dict[str, Any]) -> None:
    """Load the Case shell environment and re-exec with its Python once."""
    profile = _environment_profile(case)
    python = profile_path(profile["python"])
    marker = f"{case['case_id']}:{profile['name']}"
    if Path(sys.executable).resolve() == python.resolve() and os.environ.get("PYPTO_BENCH_ENV") == marker:
        return
    scripts = [str(profile_path(path)) for path in profile.get("setup_scripts", [])]
    command = 'set -e; for script in "$@"; do source "$script" >/dev/null; done; env -0'
    output = subprocess.check_output(["bash", "-c", command, "bench-env", *scripts], env=os.environ)
    environment = {}
    for item in output.split(b"\0"):
        if item:
            key, value = item.split(b"=", 1)
            environment[key.decode()] = value.decode()
    format_values = profile_values()
    for key, value in profile.get("prepend_env", {}).items():
        prefix = str(value).format(**format_values)
        current = environment.get(str(key), "")
        environment[str(key)] = os.pathsep.join(part for part in (prefix, current) if part)
    environment.update({str(key): str(value).format(**format_values)
                        for key, value in profile.get("env", {}).items()})
    environment["PYPTO_BENCH_ENV"] = marker
    os.execve(python, [str(python), *sys.argv], environment)


def environment(case: dict[str, Any], platform: str | None) -> tuple[dict[str, str], Any]:
    profile = _environment_profile(case)
    format_values = profile_values()
    updates = {
        str(key): str(value).format(**format_values)
        for key, value in profile.get("env", {}).items()
    }
    runtime_fields = {
        "platform": platform or profile["platform"],
        "device_id": profile.get("device", 0),
    }
    runtime_fields.update(profile.get("runtime", {}))
    runtime_fields.update(case.get("runtime", {}))
    if os.environ.get("PYPTO_BENCH_ARTIFACT_DIR"):
        runtime_fields.pop("save_kernels_dir", None)
    profile_mode = profiling_mode()
    if profile_mode != "clean":
        from profile_modes import settings

        runtime_fields.update(settings(profile_mode))
    device_ids = runtime_fields.pop("device_ids", None)
    if device_ids is not None:
        from pypto.ir.distributed_compiled_program import DistributedConfig

        runtime_fields["distributed_config"] = DistributedConfig(device_ids=device_ids)
        runtime_fields["device_id"] = device_ids[0]
    from pypto.runtime import RunConfig

    return updates, RunConfig(**runtime_fields)


def profiling_mode() -> str:
    return os.environ.get("PYPTO_BENCH_PROFILE", "clean")


@contextlib.contextmanager
def environment_scope(
    updates: dict[str, str], benchmark: bool, reset_inputs: bool = True,
    *, warmup: int = 5, rounds: int = 100,
) -> Iterator[None]:
    effective = {
        **updates,
        "PYPTO_BENCH": "1" if benchmark else "0",
        "PYPTO_BENCH_WARMUP": str(warmup),
        "PYPTO_BENCH_ROUNDS": str(rounds),
        "PYPTO_BENCH_RESET_INPUTS": "each_launch" if benchmark and reset_inputs else "none",
    }
    previous = {key: os.environ.get(key) for key in effective}
    os.environ.update(effective)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def benchmark_sizes(case: dict[str, Any]) -> tuple[int, int]:
    settings = {**_environment_profile(case).get("benchmark", {}),
                **case.get("benchmark", {})}
    return int(settings.get("warmup", 5)), int(settings.get("rounds", 10))




def bench_samples(result) -> list[float]:
    if result.bench is None:
        return []
    return [float(value) for value in result.bench.per_round("effective") if value > 0]
