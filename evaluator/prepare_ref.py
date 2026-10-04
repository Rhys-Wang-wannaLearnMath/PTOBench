#!/usr/bin/env python3
"""Dispatch one-time Ref preparation to a Case-local evaluator."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import common


def fixtures_ready(sealed: Path, case: dict) -> bool:
    fixture_ids = case.get("fixture_ids", [])
    return bool(fixture_ids) and all(
        (sealed / "fixtures" / fixture_id / "manifest.json").is_file()
        and (sealed / "fixtures" / fixture_id / "in").is_dir()
        for fixture_id in fixture_ids
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--platform")
    parser.add_argument("--regenerate-fixtures", action="store_true")
    args = parser.parse_args()
    # Ref output and its performance baseline are always clean.  Profiling is
    # a Candidate diagnostic and must never change the certified baseline.
    os.environ["PYPTO_BENCH_PROFILE"] = "clean"

    case_dir, sealed, case = common.resolve_case(args.case)
    common.bootstrap_environment(case)
    common.write_json(sealed / "certification.json", {
        "schema_version": 1, "status": "pending", "case_id": case["case_id"],
        "ref_output_cache": "pending", "cpu_oracle_role": "one_time_ref_certification",
    })
    if (sealed / "official" / "original.py").is_file() and (
        args.regenerate_fixtures or not fixtures_ready(sealed, case)
    ):
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("generate_fixtures.py")),
            case["case_id"],
        ], check=True)
    evaluator = common.load_case_evaluator(sealed)
    evaluator.prepare_ref(case_dir, sealed, case, args.platform)
    print(f"{case['case_id']}: Ref cache ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
