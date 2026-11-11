"""Shared Ref-certification and Candidate-evaluation lifecycle.

Case evaluators supply the ABI-specific adapter callbacks.  This module owns
the benchmark-wide sequence and result/cache format.
"""
from __future__ import annotations

import statistics
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

import torch

import common
import execution_record
import isolation
import timing_policy


def fixture_ids(sealed: Path, case: dict[str, Any]) -> list[str]:
    del sealed
    return common.declared_fixture_ids(case)


@contextmanager
def fixture_build_root(case, fixture_id):
    """A separate build tree prevents later inputs overwriting earlier captures."""
    previous = os.environ.get('PYPTO_PROG_BUILD_DIR')
    partition = None
    if os.environ.get('PTOBENCH_WORKER') == '1':
        partition = common.read_json(Path(os.environ['PTOBENCH_WORKER_METADATA'])).get('fixture_partition')
        if partition is not None and (partition != fixture_id or case['fixture_ids'] != [partition]):
            raise ValueError('worker fixture partition mismatch')
    if (len(common.declared_fixture_ids(case)) == 1 and partition is None) or previous is None:
        yield
        return
    os.environ['PYPTO_PROG_BUILD_DIR'] = str(Path(previous) / fixture_id)
    try:
        yield
    finally:
        os.environ['PYPTO_PROG_BUILD_DIR'] = previous


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
                   "iterations": rounds, "compiler_tuning": common.tuning_defaults(case),
                   "fixtures": {}}
    timing_policy.attach(performance, case)
    reset_inputs = case.get("state", {}).get("behavior", "none") != "none"
    resolved_tuning_by_fixture = {}
    sys.path.insert(0, str(common.BENCHMARK_ROOT))

    for fixture_id in fixture_ids(sealed, case):
        fixture = sealed / "fixtures" / fixture_id
        specs = build_specs(fixture)
        input_identity = common.fixture_input_identity(fixture)
        with fixture_build_root(case, fixture_id), common.environment_scope(
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
        resolved_tuning_by_fixture[fixture_id] = runtime._ptobench_tuning_record
        timing_policy.validate_state_abi(case, specs)
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
            "execution_contract": timing_policy.execution_contract(
                case, specs, runtime, input_identity),
            "raw_samples_effective_us": common.raw_bench_samples(result),
            "samples_effective_us": samples,
            "median_effective_us": statistics.median(samples),
            "mean_effective_us": statistics.fmean(samples),
        }

    from verify_tuning import identity as tuning_identity
    performance.update({
        "tuning_identity": tuning_identity(case_dir, case),
        "tuning_evidence": os.environ.get(
            "PTOBENCH_TUNING_EVIDENCE",
            f"{sealed.relative_to(common.sealed_root())}/ref_performance.json#fixtures",
        ),
        "resolved_tuning": (next(iter(resolved_tuning_by_fixture.values()))
                             if len(resolved_tuning_by_fixture) == 1 else None),
        "resolved_tuning_by_fixture": resolved_tuning_by_fixture,
    })
    common.write_json(sealed / "ref_performance.json", performance)
    common.write_json(sealed / "certification.json", {
        "schema_version": 1,
        "status": "ready",
        "case_id": case["case_id"],
        "ref_output_cache": "ready",
        "cpu_oracle_role": "one_time_ref_certification",
        "timing_policy": timing_policy.policy(case),
        "comparator_sha256": common.comparator_sha256(case_dir, case),
        "tuning_identity": performance["tuning_identity"],
        "tuning_evidence": performance["tuning_evidence"],
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
    make_invariant_compare: Callable[[], Any] | None = None,
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
    expected_timing_policy = timing_policy.policy(case)
    if (certification.get("timing_policy") != expected_timing_policy
            or baseline.get("timing_policy") != expected_timing_policy):
        raise RuntimeError(
            "Ref cache execution contract is stale; "
            "run maintenance prepare_ref.py --case <family.case> again"
        )
    compare = make_compare()
    tuning = common.read_tuning(candidate_workspace, case)
    updates, runtime = common.environment(case, platform, tuning=tuning)
    if isolation.enabled() and os.environ.get('PTOBENCH_WORKER') != '1':
        from candidate_execution import isolated_evaluate
        return isolated_evaluate(locals())
    if os.environ.get('PTOBENCH_WORKER') == '1':
        metadata = common.read_json(Path(os.environ['PTOBENCH_WORKER_METADATA']))
        from candidate_execution import configure_devices
        configure_devices(runtime, metadata)
        sys.path.insert(0, str(common.BENCHMARK_ROOT))
        from modes._shared.profile_environment import isolated_environment
        updates = isolated_environment(updates, metadata.get('devices'))
        from worker_startup import install, runtime_environment
        updates = runtime_environment(updates)
        install(os.environ['PTOBENCH_WORKER_METADATA'])
    program = make_program(load_candidate(), case)
    benchmark = phase in {"all", "performance"}
    reset_inputs = case.get("state", {}).get("behavior", "none") != "none"
    runs = []

    for fixture_id in fixture_ids(sealed, case):
        fixture = sealed / "fixtures" / fixture_id
        worker = os.environ.get('PTOBENCH_WORKER') == '1'
        if worker:
            from candidate_execution import worker_specs, bridge
            metadata = common.read_json(Path(os.environ['PTOBENCH_WORKER_METADATA']))
            specs = worker_specs(metadata['specs'][fixture_id], fixture)
        else:
            specs = build_specs(fixture)
        timing_policy.validate_state_abi(case, specs)
        input_identity = common.fixture_input_identity(fixture)
        warmup, rounds = common.benchmark_sizes(case)
        # Retain existing native benchmark output; add no execution or timing model.
        sys.path.insert(0, str(common.BENCHMARK_ROOT))
        from modes._shared.profile_dispatch_pairing import capture as pair_dispatches
        run_arguments = {
            "fn": program, "specs": specs, "config": runtime,
            "compare_fn": compare, "golden_data": str(fixture),
        }
        with fixture_build_root(case, fixture_id), (bridge(fixture_id) if worker else nullcontext()), execution_record.observe_runtime(runtime), common.environment_scope(
            updates, benchmark=benchmark, reset_inputs=reset_inputs,
            warmup=warmup, rounds=rounds,
        ), pair_dispatches(
                       enabled=common.profiling_mode() in ('runtime', 'combined')
                       and runtime.platform in ('a2a3','a5')):
            result = run(**run_arguments)
        # Tensor directions are stamped from the compiled artifact during run().
        # Compare the candidate contract only after that stamp exists.
        directions_ready = all(
            getattr(spec, "direction", None) is not None
            for spec in specs if hasattr(spec, "shape")
        )
        contract = None
        if directions_ready:
            contract = timing_policy.execution_contract(case, specs, runtime, input_identity)
            if baseline["fixtures"][fixture_id].get("execution_contract") != contract:
                raise RuntimeError(
                    f"Ref execution ABI/policy differs for {fixture_id}; "
                    "run maintenance prepare_ref.py --case <family.case> again"
                )
        elif result.passed:
            raise RuntimeError(f"Compiled Tensor directions are unavailable for {fixture_id}")
        build_dir = str(result.work_dir.resolve()) if result.work_dir is not None else None
        samples = common.bench_samples(result)
        if result.passed:
            timing_policy.validate_state_abi(case, specs)
        median = statistics.median(samples) if samples else None
        mean = statistics.fmean(samples) if samples else None
        ref_median = baseline["fixtures"][fixture_id]["median_effective_us"]
        ref_mean = baseline["fixtures"][fixture_id]["mean_effective_us"]
        runs.append({
            "_build_dir": build_dir,
            "fixture_id": fixture_id,
            "input_identity_sha256": input_identity,
            "passed": result.passed,
            "error": result.error or "",
            "raw_samples_effective_us": common.raw_bench_samples(result),
            "raw_samples_device_us": common.raw_bench_samples(result, 'device'),
            "measurement_config": {
                "metric": "effective", "unit": "us", "benchmark_enabled": benchmark,
                "timing_policy": expected_timing_policy,
                "execution_contract_sha256": contract["sha256"] if contract else None,
                "memory_planner": tuning["memory_planner"],
                "analyze_auto_scopes_for_deps": tuning["analyze_auto_scopes_for_deps"],
                "requested_warmup": warmup, "requested_rounds": rounds,
                "actual_warmup": getattr(result.bench, 'warmup', None),
                "actual_rounds": getattr(result.bench, 'rounds', None),
                "reset_inputs": reset_inputs,
                "tuning": getattr(runtime, "_ptobench_tuning_record", None),
            },
            "samples_effective_us": samples,
            "samples_device_us": common.bench_samples(result, 'device'),
            "median_effective_us": median,
            "mean_effective_us": mean,
            "ref_median_effective_us": ref_median,
            "ref_mean_effective_us": ref_mean,
            "speedup": ref_median / median if median else None,
            "mean_speedup": ref_mean / mean if mean else None,
        })
    return runs
