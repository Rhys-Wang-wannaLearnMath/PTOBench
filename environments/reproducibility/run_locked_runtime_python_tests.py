#!/usr/bin/env python3
"""Run runtime Python unit tests with PTOBench's locked interpreter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
ENVIRONMENTS = REPO_ROOT / "benchmarks" / "PTOBench" / "environments"


def main() -> int:
    lock = json.loads((ENVIRONMENTS / "LOCK.json").read_text(encoding="utf-8"))
    profile = json.loads((ENVIRONMENTS / "a2a3_single_card.json").read_text(encoding="utf-8"))
    python = REPO_ROOT / profile["python"]
    runtime_root = Path(lock["runtime"]["source_root"])
    if not python.is_file() or not runtime_root.is_dir():
        print("Locked PTOBench Python or runtime source is unavailable", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import pypto; from pypto.pypto_core import DataType; print(pypto.__file__)",
        ],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if probe.returncode:
        print(probe.stderr, file=sys.stderr, end="")
        return probe.returncode
    result = subprocess.run(
        [str(python), "-m", "pytest", "tests/ut/py", "-q"],
        cwd=runtime_root,
        env=env,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
