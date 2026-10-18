"""Shared Ref-certification and Candidate-evaluation lifecycle.

Case evaluators supply the ABI-specific adapter callbacks.  This module owns
the benchmark-wide sequence and result/cache format.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

import common
import execution_record


def fixture_ids(sealed: Path, case: dict[str, Any]) -> list[str]:
    del sealed
    declared = case.get("fixture_ids")
    if declared != ["fixture_000"]:
        raise ValueError("v0.2 Cases must declare exactly one fixture: fixture_000")
    return list(declared)


def prepare_ref(
    *,
    case_dir: Path,
    sealed: Path,
    case: dict[str, Any],
    platform: str | None,
    build_specs: Callable[[Path], Any],
    load_ref: Callable[[], Any],
    load_oracle: Callable[[], Any],
    make_program: Callable[[Any, dict[str, Any]], Any],
    make_compare: Callable[[], Any],
    run: Callable[..., Any],
    oracle_golden: Callable[[Any, Any], None],
) -> dict[str, Any]:
    compare = make_compare()
    updates, runtime = common.environment(case, platform)
    program = make_program(load_ref(), case)
    oracle = load_oracle()
    warmup, rounds = common.benchmark_sizes(case)
    performance = {"schema_version": 1, "status": "ready", "warmup": warmup,
                   "iterations": rounds, "compiler_tuning": dict(common.TUNING_DEFAULTS),
                   "fixtures": {}}
    reset_inputs = case.get("state", {}).get("behavior", "none") != "none"

    for fixture_id in fixture_ids(sealed, case):
        fixture = sealed / "fixtures" / fixture_id
        specs = build_specs(fixture)
        with common.environment_scope(
            updates, benchmark=True, reset_inputs=reset_inputs,
            warmup=warmup, rounds=rounds,
        ):
            result = run(
                fn=program, specs=specs, config=runtime, compare_fn=compare,
                golden_fn=lambda values: oracle_golden(oracle, values),
                capture_outputs=True,
            )
        if not result.passed:
            raise RuntimeError(f"Ref failed on {fixture_id}: {result.error}")
        outputs = result.outputs or {}
        samples = common.bench_samples(result)
        if not samples or set(outputs) != set(case["correctness"]["outputs"]):
            raise RuntimeError(f"Incomplete Ref outputs or timing on {fixture_id}")
        output_dir = fixture / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.pt"):
            stale.unlink()
        for name, value in outputs.items():
            torch.save(value, output_dir / f"{name}.pt")
        performance["fixtures"][fixture_id] = {
            "raw_samples_effective_us": common.raw_bench_samples(result),
            "samples_effective_us": samples,
            "median_effective_us": statistics.median(samples),
            "mean_effective_us": statistics.fmean(samples),
        }

    common.write_json(sealed / "ref_performance.json", performance)
    common.write_json(sealed / "certification.json", {
        "schema_version": 1,
        "status": "ready",
        "case_id": case["case_id"],
        "ref_output_cache": "ready",
        "cpu_oracle_role": "one_time_ref_certification",
        "comparator_sha256": common.comparator_sha256(case_dir, case),
    })
    return performance


def evaluate_candidate(
    *,
    case_dir: Path,
    sealed: Path,
    case: dict[str, Any],
    candidate_workspace: Path,
    platform: str | None,
    phase: str,
    build_specs: Callable[[Path], Any],
    load_candidate: Callable[[], Any],
    make_program: Callable[[Any, dict[str, Any]], Any],
    make_compare: Callable[[], Any],
    run: Callable[..., Any],
) -> list[dict[str, Any]]:
    certification = common.read_json(sealed / "certification.json")
    if certification.get("status") != "ready":
        raise RuntimeError("Ref cache is not ready; run maintenance prepare_ref.py --case <family.case> first")
    expected_comparator_sha256 = common.comparator_sha256(case_dir, case)
    if certification.get("comparator_sha256") != expected_comparator_sha256:
        raise RuntimeError(
            "Ref cache certification does not match public comparator; "
            "run maintenance prepare_ref.py --case <family.case> again"
        )
    baseline = common.read_json(sealed / "ref_performance.json")
    compare = make_compare()
    tuning = common.read_tuning(candidate_workspace)
    updates, runtime = common.environment(case, platform, tuning=tuning)
    program = make_program(load_candidate(), case)
    benchmark = phase in {"all", "performance"}
    reset_inputs = case.get("state", {}).get("behavior", "none") != "none"
    runs = []

    for fixture_id in fixture_ids(sealed, case):
        fixture = sealed / "fixtures" / fixture_id
        specs = build_specs(fixture)
        warmup, rounds = common.benchmark_sizes(case)
        with execution_record.observe_runtime(runtime), common.environment_scope(
            updates, benchmark=benchmark, reset_inputs=reset_inputs,
            warmup=warmup, rounds=rounds,
        ):
            result = run(
                fn=program, specs=specs, config=runtime, compare_fn=compare,
                golden_data=str(fixture),
            )
        build_dir = str(result.work_dir.resolve()) if result.work_dir is not None else None
        samples = common.bench_samples(result)
        median = statistics.median(samples) if samples else None
        mean = statistics.fmean(samples) if samples else None
        ref_median = baseline["fixtures"][fixture_id]["median_effective_us"]
        ref_mean = baseline["fixtures"][fixture_id]["mean_effective_us"]
        runs.append({
            "_build_dir": build_dir,
            "fixture_id": fixture_id,
            "passed": result.passed,
            "error": result.error or "",
            "raw_samples_effective_us": common.raw_bench_samples(result),
            "measurement_config": {
                "metric": "effective", "unit": "us", "benchmark_enabled": benchmark,
                "memory_planner": tuning["memory_planner"],
                "analyze_auto_scopes_for_deps": tuning["analyze_auto_scopes_for_deps"],
                "requested_warmup": warmup, "requested_rounds": rounds,
                "actual_warmup": getattr(result.bench, 'warmup', None),
                "actual_rounds": getattr(result.bench, 'rounds', None),
                "reset_inputs": reset_inputs,
            },
            "samples_effective_us": samples,
            "median_effective_us": median,
            "mean_effective_us": mean,
            "ref_median_effective_us": ref_median,
            "ref_mean_effective_us": ref_mean,
            "speedup": ref_median / median if median else None,
            "mean_speedup": ref_mean / mean if mean else None,
        })
    return runs
