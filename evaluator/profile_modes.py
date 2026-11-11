"""Profiling modes and the modes exposed to newly initialized workspaces."""

from __future__ import annotations

PROFILE_MODES = {
    "runtime": {
        "description": "Runtime scheduling, dependency, and scope diagnostics.",
        "phase": "correctness",
        "run_config": {
            "enable_chip_swimlane": 4,
            "enable_dep_gen": True,
            "enable_scope_stats": True,
        },
    },
    "hardware": {
        "description": "Hardware PMU diagnostics.",
        # PMU collection is a one-shot diagnostic. Reusing the profiled handle
        # for benchmark warmups/rounds fails on the current runtime.
        "phase": "correctness",
        "run_config": {"enable_pmu": 2},
    },
    "compile": {
        "description": "Compiler pass IR and compile-time diagnostics.",
        "phase": "correctness",
        "run_config": {
            "dump_passes": True,
            "compile_profiling": True,
        },
    },
}

PROFILE_STRATEGIES = ("split", "combined")
DEFAULT_PROFILE_STRATEGY = "combined"
COMBINED_PROFILE_MODE = "combined"

# The current profile table is the single source of truth.  Deriving the
# enabled tuple prevents an older mode list from surviving beside it.
ENABLED_PROFILE_MODES = tuple(PROFILE_MODES)
PUBLIC_TO_INTERNAL = {"runtime": "runtime", "pmu": "hardware", "compiler": "compile"}
COMBINED_PUBLIC_TO_INTERNAL = {"combined": COMBINED_PROFILE_MODE}


def normalize_strategy(strategy: str | None) -> str:
    value = DEFAULT_PROFILE_STRATEGY if strategy is None else strategy
    if value not in PROFILE_STRATEGIES:
        raise ValueError(f"unsupported profiling strategy: {value}")
    return value


def public_enabled_modes(strategy: str | None = None) -> dict[str, dict[str, str]]:
    strategy = normalize_strategy(strategy)
    enabled = enabled_modes()
    if strategy == "combined":
        return ({"combined": {"description": "Runtime, PMU, and compiler diagnostics."}}
                if enabled else {})
    return {public: enabled[internal] for public, internal in PUBLIC_TO_INTERNAL.items()
            if public != "combined" and internal in enabled}


def enabled_modes() -> dict[str, dict[str, str]]:
    return {
        name: {"description": PROFILE_MODES[name]["description"]}
        for name in ENABLED_PROFILE_MODES
    }


def settings(mode: str | None) -> dict:
    if mode is None or mode == "clean":
        return {}
    if mode == COMBINED_PROFILE_MODE:
        combined = {}
        for name in ENABLED_PROFILE_MODES:
            combined.update(PROFILE_MODES[name]["run_config"])
        return combined
    try:
        return dict(PROFILE_MODES[mode]["run_config"])
    except KeyError as exc:
        raise ValueError(f"unsupported profiling mode: {mode}") from exc


def phase(mode: str) -> str:
    if mode == COMBINED_PROFILE_MODE:
        return "correctness"
    try:
        return str(PROFILE_MODES[mode]["phase"])
    except KeyError as exc:
        raise ValueError(f"unsupported profiling mode: {mode}") from exc


def validate_artifacts(build_dir, mode):
    """Require the files requested by this profile mode to be present."""
    if mode == COMBINED_PROFILE_MODE:
        return {
            "schema_version": 1,
            "mode": mode,
            "status": "artifacts_present",
            "artifacts": sum(
                (validate_artifacts(build_dir, child)["artifacts"]
                 for child in ENABLED_PROFILE_MODES), []),
        }
    from pathlib import Path
    build = Path(build_dir)
    if not build.is_dir():
        raise RuntimeError('diagnostic build directory missing')
    patterns = {
        'runtime': (),
        'hardware': ('pmu.csv',),
        'compile': ('pipeline_profile.json', 'perf_hints.log'),
    }
    try:
        required = patterns[mode]
    except KeyError as exc:
        raise ValueError(f'unsupported profiling mode: {mode}') from exc

    present = []
    if mode == 'runtime':
        records = sorted(build.rglob('chip_swimlane_records.json'))
        if not records:
            raise RuntimeError('runtime capture incomplete: chip_swimlane_records.json missing')
        for record in records:
            graph = record.parent / 'deps.json'
            if not graph.is_file() or graph.stat().st_size == 0:
                raise RuntimeError('runtime capture incomplete: paired deps.json missing or empty')
            present.extend((str(record.relative_to(build)), str(graph.relative_to(build))))

    for pattern in required:
        paths = sorted(build.rglob(pattern))
        if not paths:
            raise RuntimeError(f'diagnostic capture incomplete: {pattern} missing')
        if pattern != 'perf_hints.log' and any(not path.stat().st_size for path in paths):
            raise RuntimeError(f'diagnostic capture incomplete: {pattern} empty')
        present.extend(str(path.relative_to(build)) for path in paths)
    if mode == 'compile' and not (
        list(build.rglob('memory_after_AllocateMemoryAddr.txt'))
        or list(build.rglob('*_after_AllocateMemoryAddr.py'))
    ):
        raise RuntimeError('compiler capture incomplete: allocated memory IR/report missing')
    return {'schema_version': 1, 'mode': mode, 'status': 'artifacts_present',
            'artifacts': present}
