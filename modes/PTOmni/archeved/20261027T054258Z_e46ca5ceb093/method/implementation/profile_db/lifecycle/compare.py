# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Neutral before/after comparison with a compatibility gate (DESIGN.md
6.5/8.2). Two runs are comparable only when their program, swimlane
level, clock, core count, and core topology all match (the capture-side
compatibility gate); anything else is refused with a reason. Deltas are
plain arithmetic over stored measurements — this module never judges
whether a change is a win or a regression."""

from __future__ import annotations

import json
from typing import Any, Mapping

from profile_db.errors import LifecycleError

_METRIC_COLS = (
    "bench_mean_us",
    "makespan_us",
    "raw_span_us",
    "cpm_us",
)


def _run_meta(conn, run_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT program, swimlane_level, clock_freq_hz, num_cores, "
        "CAST(core_types AS VARCHAR), bench_mean_us, makespan_us, raw_span_us, cpm_us "
        "FROM run WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if row is None:
        return None
    try:
        core_types = json.loads(row[4] or "[]")
    except (TypeError, ValueError):
        core_types = []
    return {
        "program": row[0],
        "swimlane_level": row[1],
        "clock_freq_hz": row[2],
        "num_cores": row[3],
        "core_types": core_types if isinstance(core_types, list) else [],
        "bench_mean_us": row[5],
        "makespan_us": row[6],
        "raw_span_us": row[7],
        "cpm_us": row[8],
    }


def _counts(conn, run_id: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, table in (("tasks", "task"), ("task_rows", "task_row"), ("edges", "dep_edge")):
        counts[name] = int(
            conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", [run_id]).fetchone()[0]
        )
    return counts


def compat_reasons(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """The reasons two runs cannot be compared (empty = comparable)."""
    reasons: list[str] = []
    for key in ("program", "swimlane_level", "clock_freq_hz", "num_cores"):
        if a.get(key) != b.get(key):
            reason = f"{key} differs: {a.get(key)!r} vs {b.get(key)!r}"
            if key == "program":
                # The default program name comes from the name_map filename,
                # which carries the one-shot build-dir hash — two captures of
                # one kernel differ unless ingest normalized them.
                reason += (
                    " (two captures of one kernel compare only after re-ingesting"
                    " both with the same --program)"
                )
            reasons.append(reason)
    if list(a.get("core_types") or []) != list(b.get("core_types") or []):
        reasons.append(
            f"core_types differs: {list(a.get('core_types') or [])} vs {list(b.get('core_types') or [])}"
        )
    return reasons


def _family_stats(conn, run_id: int) -> dict[str, dict[str, float]]:
    """family -> {count, busy_us, wall_us} for one run."""
    rows = conn.execute(
        "SELECT family, COUNT(*), COALESCE(SUM(busy_us), 0), "
        "COALESCE(SUM(wall_us), 0) FROM task "
        "WHERE run_id = ? AND family IS NOT NULL GROUP BY family",
        [run_id],
    ).fetchall()
    return {
        str(family): {
            "count": float(count),
            "busy_us": float(busy),
            "wall_us": float(wall),
        }
        for family, count, busy, wall in rows
    }


def _family_deltas(
    stats_a: Mapping[str, Mapping[str, float]],
    stats_b: Mapping[str, Mapping[str, float]],
    family: str,
) -> list[dict[str, Any]]:
    names = set(stats_a) | set(stats_b)
    if family != "*":
        names = {family} if family in names else set()
    ranked = sorted(
        names,
        key=lambda name: (
            -abs(stats_b.get(name, {}).get("busy_us", 0.0) - stats_a.get(name, {}).get("busy_us", 0.0)),
            name,
        ),
    )
    deltas: list[dict[str, Any]] = []
    empty = {"count": 0.0, "busy_us": 0.0, "wall_us": 0.0}
    for name in ranked:
        before = stats_a.get(name, empty)
        after = stats_b.get(name, empty)
        for metric, key in (
            ("family_busy_us", "busy_us"),
            ("family_wall_us", "wall_us"),
            ("family_tasks", "count"),
        ):
            b, a = before[key], after[key]
            deltas.append(
                {
                    "metric": metric,
                    "family": name,
                    "before": b,
                    "after": a,
                    "delta": a - b,
                    "ratio": (a / b) if b else None,
                }
            )
    return deltas


def compare_runs(conn, run_a: int, run_b: int, *, family: str | None = None) -> dict[str, Any]:
    """Compare two runs. Returns ``{run_a, run_b, program, compatible,
    reasons, deltas}``. Raises ``LifecycleError`` when incompatible.

    ``family`` is ``None`` (run-level only), ``"*"`` (every family), or a
    single family name.
    """
    meta_a = _run_meta(conn, run_a)
    meta_b = _run_meta(conn, run_b)
    if meta_a is None:
        raise LifecycleError(f"run {run_a} does not exist")
    if meta_b is None:
        raise LifecycleError(f"run {run_b} does not exist")

    from profile_db.runtime_adapter.storage import read_snapshot, semantic_identity
    from profile_db.runtime_adapter import ANALYSIS_VERSION

    def semantics(run_id):
        docs = [read_snapshot(conn, run_id, family) for family in ("critical_path", "scheduler")]
        if any(d.get("analysis_version") != ANALYSIS_VERSION for d in docs):
            return None
        return [semantic_identity(d) for d in docs]

    reasons = compat_reasons(meta_a, meta_b)
    semantic_a, semantic_b = semantics(run_a), semantics(run_b)
    profile_reasons = list(reasons)
    if semantic_a is None or semantic_b is None or semantic_a != semantic_b:
        profile_reasons.append("analysis semantics differ or are stale; refresh both runs")
    if reasons:
        raise LifecycleError(f"runs {run_a} and {run_b} are not comparable: " + "; ".join(reasons))

    counts_a = _counts(conn, run_a)
    counts_b = _counts(conn, run_b)

    deltas: list[dict[str, Any]] = []
    for metric in _METRIC_COLS:
        if profile_reasons and not metric.startswith("bench_"):
            continue
        before = meta_a.get(metric)
        after = meta_b.get(metric)
        if before is None or after is None:
            continue
        deltas.append(
            {
                "metric": metric,
                "before": before,
                "after": after,
                "delta": after - before,
                "ratio": (after / before) if before else None,
            }
        )
    for name in () if profile_reasons else ("tasks", "task_rows", "edges"):
        before = counts_a[name]
        after = counts_b[name]
        deltas.append(
            {
                "metric": name,
                "before": before,
                "after": after,
                "delta": after - before,
                "ratio": (after / before) if before else None,
            }
        )
    if family is not None and not profile_reasons:
        deltas.extend(_family_deltas(_family_stats(conn, run_a), _family_stats(conn, run_b), family))
    return {
        "run_a": run_a,
        "run_b": run_b,
        "program": meta_a["program"],
        "compatible": not profile_reasons,
        "profile_compatible": not profile_reasons,
        "reasons": profile_reasons,
        "deltas": deltas,
    }
