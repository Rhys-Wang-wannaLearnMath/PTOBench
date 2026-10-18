"""Small shared helpers for v0.2 Case evaluators."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
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


TUNING_DEFAULTS = {
    "memory_planner": "PYPTO",
    "analyze_auto_scopes_for_deps": False,
}
TUNING_ALLOWED = {
    "memory_planner": frozenset(("PYPTO", "DSA_RP", "PTOAS")),
    "analyze_auto_scopes_for_deps": (bool,),
}


def read_tuning(workspace: Path) -> dict[str, Any]:
    """Read the small, public Candidate tuning contract.

    The file is intentionally outside ``.benchmark_context`` so an Agent can
    edit it. Missing fields use the public defaults; unknown fields and values
    are rejected before any Candidate code is imported or run.
    """
    path = Path(workspace) / "tuning.json"
    if not path.exists():
        return dict(TUNING_DEFAULTS)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid tuning.json: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("tuning.json must contain an object")
    unknown = set(value) - set(TUNING_DEFAULTS)
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"unknown tuning option(s): {names}")
    result = dict(TUNING_DEFAULTS)
    result.update(value)
    planner = result["memory_planner"]
    if not isinstance(planner, str) or planner not in TUNING_ALLOWED["memory_planner"]:
        allowed = ", ".join(sorted(TUNING_ALLOWED["memory_planner"]))
        raise ValueError(f"memory_planner must be one of: {allowed}")
    analyze = result["analyze_auto_scopes_for_deps"]
    if type(analyze) is not bool:
        raise ValueError("analyze_auto_scopes_for_deps must be true or false")
    return result


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


def case_relative_path(case_id: str) -> Path:
    """Return the canonical storage path for a Case identifier."""
    family, name = str(case_id).split(".", 1)
    return Path(family) / name


def case_paths(case_id: str) -> tuple[Path, Path, dict[str, Any]]:
    """Locate a Case and its sealed assets at their canonical paths."""
    relative = case_relative_path(case_id)
    case_dir = BENCHMARK_ROOT / "cases" / relative
    case = read_json(case_dir / "case.json")
    sealed = sealed_root() / relative
    return case_dir.resolve(), sealed.resolve(), case


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
    # Publish complete records atomically. Imported historical assets can share
    # inodes, so replacing this path also avoids changing an archived record.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        case = read_json(case_dir / "case.json")
        relative = case_dir.relative_to(BENCHMARK_ROOT / "cases")
        sealed = sealed_root() / relative
    else:
        case_dir, sealed, case = case_paths(str(value))
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
    path, name, seed, *, config_alias=None,
    builder_name="build_tensor_specs", builder_args=(), builder_kwargs=None,
):
    """Construct historical fixture specs without changing preserved source.

    Legacy output membership is construction metadata only. Once the runtime
    stamps a direction, the current compiled artifact remains authoritative.
    Historical run_jit is importable here, but cannot launch a workload.
    """
    import golden
    import torch

    previous_config = sys.modules.get("config")
    original_tensor_spec = golden.TensorSpec
    missing = object()
    original_run_jit = getattr(golden, "run_jit", missing)

    class FixtureTensorSpec(original_tensor_spec):
        def __init__(self, *args, is_output=False, **kwargs):
            super().__init__(*args, **kwargs)
            self._fixture_legacy_output = bool(is_output)

        @property
        def is_output(self):
            if self.direction is None:
                return self._fixture_legacy_output
            return super().is_output

    def forbidden_historical_run(*args, **kwargs):
        raise RuntimeError("historical run_jit cannot execute during Fixture construction")

    with _isolated_module_scope((path.parent,)):
        previous_path = sys.path[:]
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.path.insert(0, str(path.parent))
        sys.dont_write_bytecode = True
        if config_alias is not None:
            sys.modules["config"] = config_alias
        golden.TensorSpec = FixtureTensorSpec
        if original_run_jit is missing:
            golden.run_jit = forbidden_historical_run
        try:
            official = load_module(path, name)
            torch.manual_seed(seed)
            builder = getattr(official, builder_name)
            return builder(*builder_args, **(builder_kwargs or {}))
        finally:
            golden.TensorSpec = original_tensor_spec
            if original_run_jit is missing:
                del golden.run_jit
            else:
                golden.run_jit = original_run_jit
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
    # Evaluator callbacks may import public config/helpers lazily, after the
    # evaluator itself has been imported. Keep that Case's import scope alive
    # for the complete call, without leaking another Case's config module.
    relative = sealed.relative_to(sealed_root())
    public = BENCHMARK_ROOT / "cases" / relative / "public"
    directories = (public, sealed)
    with local_import_scope(directories):
        module = load_module(sealed / "evaluate.py", f"case_evaluator_{sealed.name.replace('.', '_')}")

    def scoped(callback):
        @functools.wraps(callback)
        def invoke(*args, **kwargs):
            with local_import_scope(directories):
                return callback(*args, **kwargs)
        return invoke

    for name in ("prepare_ref", "evaluate_candidate"):
        if hasattr(module, name):
            setattr(module, name, scoped(getattr(module, name)))
    return module


@contextlib.contextmanager
def local_import_scope(directories: tuple[Path, ...]) -> Iterator[None]:
    """Support lazy local imports while restoring paths and colliding modules."""
    previous_path = sys.path[:]
    previous_bytecode = sys.dont_write_bytecode
    with _isolated_module_scope(directories):
        sys.path[:0] = [str(directory) for directory in directories]
        sys.dont_write_bytecode = True
        try:
            yield
        finally:
            sys.path[:] = previous_path
            sys.dont_write_bytecode = previous_bytecode


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


def configured_environment(profile: dict[str, Any], base: dict[str, str]) -> dict[str, str]:
    """Apply profile exports and the installed NPU toolchain's C++ runtime."""
    environment = dict(base)
    format_values = profile_values()
    for key, value in profile.get("prepend_env", {}).items():
        prefix = str(value).format(**format_values)
        current = environment.get(str(key), "")
        environment[str(key)] = os.pathsep.join(part for part in (prefix, current) if part)
    environment.update({str(key): str(value).format(**format_values)
                        for key, value in profile.get("env", {}).items()})
    # PTOAS v0.60 needs GLIBCXX_3.4.29. Use the already installed runtime for
    # every NPU profile, including compiler subprocesses spawned by Agent runs.
    # This changes library resolution only; no packages are installed/upgraded.
    miniforge = format_values["miniforge_root"]
    if profile.get("platform") != "cpu" and miniforge:
        library_dir = Path(miniforge) / "lib"
        if (library_dir / "libstdc++.so.6").is_file():
            entries = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                [str(library_dir), *(p for p in entries if p and p != str(library_dir))]
            )
    return environment


def bootstrap_environment(case: dict[str, Any]) -> None:
    """Load the Case shell environment and re-exec with its Python once."""
    profile = _environment_profile(case)
    python = profile_path(profile["python"])
    marker = f"{case['case_id']}:{profile['name']}:cxx-runtime-v1"
    if Path(sys.executable).resolve() == python.resolve() and os.environ.get("PYPTO_BENCH_ENV") == marker:
        validate_toolchain(profile)
        return
    environment = load_profile_environment(profile)
    environment["PYPTO_BENCH_ENV"] = marker
    os.execve(python, [str(python), *sys.argv], environment)


def load_profile_environment(profile: dict[str, Any]) -> dict[str, str]:
    """Load exactly the shell exports used by both Launcher and Evaluator."""
    scripts = [str(profile_path(path)) for path in profile.get("setup_scripts", [])]
    command = 'set -e; for script in "$@"; do source "$script" >/dev/null; done; env -0'
    output = subprocess.check_output(["bash", "-c", command, "bench-env", *scripts], env=os.environ)
    environment = {}
    for item in output.split(b"\0"):
        if item:
            key, value = item.split(b"=", 1)
            environment[key.decode()] = value.decode()
    return configured_environment(profile, environment)


def validate_toolchain(profile: dict[str, Any], environment: dict[str, str] | None = None) -> None:
    """Fail before certification/Agent execution if the compiler cannot load."""
    if profile.get("platform") == "cpu":
        return
    exports = os.environ if environment is None else environment
    root = exports.get("PTOAS_ROOT") or profile_values()["ptoas_root"]
    compiler = str(Path(root) / "bin" / "ptoas") if root else "ptoas"
    result = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, timeout=30,
        env=environment,
    )
    if result.returncode:
        raise RuntimeError(
            f"PTOAS environment preflight failed ({compiler}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def environment(
    case: dict[str, Any], platform: str | None,
    tuning: dict[str, Any] | None = None,
) -> tuple[dict[str, str], Any]:
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
    # These compiler options are controlled by the Candidate workspace. The
    # default is also used by Ref preparation, so historical Case runtime
    # overrides cannot silently change the certified baseline.
    effective_tuning = dict(TUNING_DEFAULTS if tuning is None else tuning)
    if os.environ.get("PYPTO_BENCH_ARTIFACT_DIR"):
        runtime_fields.pop("save_kernels_dir", None)
    profile_mode = profiling_mode()
    if profile_mode != "clean":
        from profile_modes import settings

        runtime_fields.update(settings(profile_mode))
    # Apply Agent tuning last so diagnostic presets cannot silently replace it.
    runtime_fields.update(effective_tuning)
    device_ids = runtime_fields.pop("device_ids", None)
    if device_ids is not None:
        from pypto.ir.distributed_compiled_program import DistributedConfig

        runtime_fields["distributed_config"] = DistributedConfig(device_ids=device_ids)
        runtime_fields["device_id"] = device_ids[0]
    from pypto.pypto_core.passes import MemoryPlanner

    runtime_fields["memory_planner"] = MemoryPlanner[effective_tuning["memory_planner"]]
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
    import math
    return [value for value in raw_bench_samples(result)
            if isinstance(value, float) and math.isfinite(value) and value > 0]


def raw_bench_samples(result) -> list:
    """Preserve timing slots in order; nonfinite values use explicit JSON strings."""
    import math
    if result.bench is None:
        return []
    samples = []
    for value in result.bench.per_round('effective'):
        if value is None:
            samples.append(None)
        else:
            value = float(value)
            samples.append(value if math.isfinite(value) else str(value))
    return samples
