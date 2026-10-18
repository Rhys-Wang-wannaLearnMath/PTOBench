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
        "run_config": {
            "enable_pmu": 2,
        },
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

# The current profile table is the single source of truth.  Deriving the
# enabled tuple prevents an older mode list from surviving beside it.
ENABLED_PROFILE_MODES = tuple(PROFILE_MODES)
PUBLIC_TO_INTERNAL = {"runtime": "runtime", "pmu": "hardware", "compiler": "compile"}


def public_enabled_modes() -> dict[str, dict[str, str]]:
    enabled = enabled_modes()
    return {public: enabled[internal] for public, internal in PUBLIC_TO_INTERNAL.items()
            if internal in enabled}


def enabled_modes() -> dict[str, dict[str, str]]:
    return {
        name: {"description": PROFILE_MODES[name]["description"]}
        for name in ENABLED_PROFILE_MODES
    }


def settings(mode: str | None) -> dict:
    if mode is None or mode == "clean":
        return {}
    try:
        return dict(PROFILE_MODES[mode]["run_config"])
    except KeyError as exc:
        raise ValueError(f"unsupported profiling mode: {mode}") from exc


def phase(mode: str) -> str:
    try:
        return str(PROFILE_MODES[mode]["phase"])
    except KeyError as exc:
        raise ValueError(f"unsupported profiling mode: {mode}") from exc
