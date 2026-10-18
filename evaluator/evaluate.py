#!/usr/bin/env python3
"""Dispatch Candidate evaluation and write a uniform result."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import common
import workspace as workspace_tools
import execution_record
import profile_execution


def prepare_run(
    candidate_workspace: Path, profile_mode: str, requested_result: Path | None,
) -> tuple[Path, Path]:
    artifacts = workspace_tools.artifact_dir(candidate_workspace, profile_mode)
    artifacts.mkdir(parents=True, exist_ok=True)
    result_path = requested_result.resolve() if requested_result else artifacts / "latest_result.json"
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(artifacts / "build_output")
    os.environ["PYPTO_BENCH_ARTIFACT_DIR"] = str(artifacts)
    os.chdir(candidate_workspace)
    return artifacts, result_path


def main() -> int:
    common.await_action_start()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--platform")
    parser.add_argument("--phase", choices=("correctness", "performance", "all"), default="all")
    parser.add_argument(
        "--profile-mode", choices=("clean", "runtime", "hardware", "compile"),
        default="clean",
    )
    parser.add_argument(
        "--measurement-mode", choices=("formal", "exploration"), default="formal",
        help="Exploration uses 2 warmups and 3 measured launches; formal uses Case settings.",
    )
    parser.add_argument("--result", type=Path)
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Internal Agent profile policy; not valid for clean evaluation.")
    args = parser.parse_args()
    if args.skip_correctness and (args.profile_mode == "clean" or args.phase != "correctness"):
        parser.error("--skip-correctness requires a one-shot profiling mode")
    # The ordinary evaluator is always clean unless the profiling command
    # explicitly requests the fixed runtime mode.
    os.environ["PYPTO_BENCH_PROFILE"] = args.profile_mode

    case_dir, sealed, case = common.resolve_case(args.case)
    common.bootstrap_environment(case)
    if args.measurement_mode == "exploration":
        # Override only this invocation, leaving canonical Case/Ref settings intact.
        case = {**case, "benchmark": {"warmup": 2, "rounds": 3}}
    workspace = (args.candidate or case_dir / case["workspace"]).resolve()
    artifacts, result_path = prepare_run(workspace, args.profile_mode, args.result)
    tuning = common.read_tuning(workspace)
    execution_path = result_path.parent / 'execution.json'
    os.environ['PTOBENCH_EXECUTION_RECORD'] = str(execution_path)
    lifecycle = {'schema_version': 1, 'started_at': execution_record.now(),
                 'platform': args.platform or common._environment_profile(case)['platform'],
                 'device': common._environment_profile(case).get('device'),
                 'phase': args.phase, 'profiling_mode': args.profile_mode,
                 'events': []}
    execution_record.write(execution_path, lifecycle)
    try:
        evaluator = common.load_case_evaluator(sealed)
        with profile_execution.output_validation(enabled=not args.skip_correctness):
            runs = evaluator.evaluate_candidate(
                case_dir, sealed, case, workspace, args.platform, args.phase,
            )
    except BaseException as error:
        lifecycle = common.read_json(execution_path)
        lifecycle.update(exception_type=type(error).__name__, message=str(error))
        execution_record.write(execution_path, lifecycle)
        raise
    finally:
        lifecycle = common.read_json(execution_path)
        lifecycle['ended_at'] = execution_record.now()
        execution_record.write(execution_path, lifecycle)
        os.environ.pop('PTOBENCH_EXECUTION_RECORD', None)
    build_dir = None
    for run in runs:
        build_dir = run.pop("_build_dir", build_dir)
    if args.profile_mode != "clean":
        for run in runs:
            run.pop("ref_median_effective_us", None)
            run.pop("ref_mean_effective_us", None)
            run.pop("speedup", None)
            run.pop("mean_speedup", None)
    summary = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "candidate": str(workspace),
        "artifact_dir": str(artifacts),
        "build_dir": build_dir,
        "phase": args.phase,
        "profiling_mode": args.profile_mode,
        "effective_tuning": tuning,
        "official_score": args.profile_mode == "clean",
        "diagnostic_only": args.profile_mode != "clean",
        "verdict": "PASS" if all(run["passed"] for run in runs) else "FAIL",
        "runs": runs,
    }
    common.write_json(result_path, summary)
    print(f"{summary['case_id']}: {summary['verdict']}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
