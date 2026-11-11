# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Named baselines (DESIGN.md 8.2).

A baseline names a run plus its unprofiled benchmark mean (the
``PYPTO_BENCH`` number, distinct from profiled makespan) and acceptance
criteria. ``diff_baseline`` compares a run against a baseline's run using
the same compatibility gate as ``compare`` — the only long-term value of
retained historical data is relative comparison.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from profile_db.errors import LifecycleError
from profile_db.lifecycle.compare import compare_runs
from profile_db.lifecycle.ids import next_id


def add_baseline(
    conn,
    *,
    name: str,
    run_id: int,
    bench_mean_us: float | None = None,
    criteria: Mapping[str, Any] | None = None,
) -> int:
    """Register a named baseline for a run; returns its baseline_id. When
    ``bench_mean_us`` is omitted, the run's stored value is used."""
    run = conn.execute(
        "SELECT program, platform, bench_mean_us FROM run WHERE run_id = ?", [run_id]
    ).fetchone()
    if run is None:
        raise LifecycleError(f"run {run_id} does not exist")
    if bench_mean_us is None:
        bench_mean_us = run[2]
    baseline_id = next_id(conn, "baseline", "baseline_id")
    conn.execute(
        "INSERT INTO baseline (baseline_id, name, program, platform, run_id, "
        "bench_mean_us, criteria, accepted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, CAST(? AS JSON), CURRENT_TIMESTAMP)",
        [
            baseline_id,
            name,
            run[0],
            run[1],
            run_id,
            bench_mean_us,
            json.dumps(dict(criteria or {})),
        ],
    )
    return baseline_id


def list_baselines(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT baseline_id, name, program, platform, run_id, bench_mean_us, "
        "CAST(criteria AS VARCHAR) FROM baseline ORDER BY baseline_id"
    ).fetchall()
    return [
        {
            "baseline_id": r[0],
            "name": r[1],
            "program": r[2],
            "platform": r[3],
            "run_id": r[4],
            "bench_mean_us": r[5],
            "criteria": json.loads(r[6] or "{}"),
        }
        for r in rows
    ]


def _resolve_baseline(conn, name: str | None) -> dict[str, Any] | None:
    baselines = list_baselines(conn)
    if not baselines:
        return None
    if name is None:
        return baselines[-1]  # latest accepted baseline
    for baseline in baselines:
        if baseline["name"] == name:
            return baseline
    return None


def diff_baseline(conn, run_id: int, baseline_name: str | None = None) -> dict[str, Any]:
    """Compare ``run_id`` against a baseline's run (compat-gated)."""
    baseline = _resolve_baseline(conn, baseline_name)
    if baseline is None:
        if baseline_name is None:
            raise LifecycleError("no baseline registered")
        raise LifecycleError(f"baseline {baseline_name!r} does not exist")
    if baseline["run_id"] is None:
        raise LifecycleError(f"baseline {baseline_name!r} references no run")
    comparison = compare_runs(conn, baseline["run_id"], run_id)
    comparison["baseline"] = baseline["name"]
    comparison["baseline_run_id"] = baseline["run_id"]
    return comparison
