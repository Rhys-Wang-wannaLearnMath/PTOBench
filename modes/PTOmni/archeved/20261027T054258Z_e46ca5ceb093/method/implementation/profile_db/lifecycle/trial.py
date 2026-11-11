# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Short-term trial memory (DESIGN.md 8.2).

A trial is one tuning experiment: ``register_trial`` opens it (running,
pending verdict), ingest binds a run to it via ``bind_trial``, and
``set_verdict`` closes it. ``parent_trial_id`` links experiments into a
lineage tree. Trial rows are memory and survive pruning.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from profile_db.errors import LifecycleError
from profile_db.lifecycle.ids import next_id

_STATUS = ("running", "done", "abandoned")
_VERDICT = ("win", "neutral", "regression", "compile_error", "pending")


def register_trial(
    conn,
    *,
    goal: str,
    hypothesis: str,
    changed_files: Sequence[str] = (),
    parent_trial_id: int | None = None,
) -> int:
    """Create a trial in the ``running`` state; returns its trial_id."""
    trial_id = next_id(conn, "trial", "trial_id")
    conn.execute(
        "INSERT INTO trial (trial_id, parent_trial_id, run_id, goal, hypothesis, "
        "changed_files, status, verdict, evidence_refs, created_at, notes) "
        "VALUES (?, ?, NULL, ?, ?, CAST(? AS JSON), 'running', 'pending', "
        "CAST('[]' AS JSON), CURRENT_TIMESTAMP, NULL)",
        [trial_id, parent_trial_id, goal, hypothesis, json.dumps(list(changed_files))],
    )
    return trial_id


def bind_trial(conn, trial_id: int, run_id: int) -> None:
    """Attach an ingested run to a trial (the experiment's evidence)."""
    if conn.execute("SELECT 1 FROM trial WHERE trial_id = ?", [trial_id]).fetchone() is None:
        raise LifecycleError(f"trial {trial_id} does not exist")
    # An active trial's run is prune-protected, so a bogus id would create a
    # phantom protection and a dangling reference in `trials list`.
    if conn.execute("SELECT 1 FROM run WHERE run_id = ?", [run_id]).fetchone() is None:
        raise LifecycleError(f"run {run_id} does not exist; ingest the capture first")
    conn.execute("UPDATE trial SET run_id = ? WHERE trial_id = ?", [run_id, trial_id])


def set_verdict(
    conn, trial_id: int, verdict: str, evidence_refs: Sequence[Any] = ()
) -> None:
    """Close a trial with its verdict.

    ``win`` / ``neutral`` / ``regression`` close a measured experiment.
    ``compile_error`` closes a trial that never produced a run (no bind
    required). ``pending`` is the open-state marker and is refused here.
    """
    if verdict not in _VERDICT or verdict == "pending":
        raise LifecycleError(
            f"invalid verdict {verdict!r}; use win/neutral/regression/compile_error"
        )
    if conn.execute("SELECT 1 FROM trial WHERE trial_id = ?", [trial_id]).fetchone() is None:
        raise LifecycleError(f"trial {trial_id} does not exist")
    conn.execute(
        "UPDATE trial SET verdict = ?, status = 'done', evidence_refs = CAST(? AS JSON) "
        "WHERE trial_id = ?",
        [verdict, json.dumps(list(evidence_refs)), trial_id],
    )


def attach_bench(conn, trial_id: int, bench: Mapping[str, Any]) -> None:
    """Store unprofiled bench summary numbers on a trial (no run required)."""
    if conn.execute("SELECT 1 FROM trial WHERE trial_id = ?", [trial_id]).fetchone() is None:
        raise LifecycleError(f"trial {trial_id} does not exist")
    conn.execute(
        "UPDATE trial SET bench_min_us = ?, bench_median_us = ?, bench_mean_us = ?, "
        "bench_max_us = ?, bench_rounds = ? WHERE trial_id = ?",
        [
            bench.get("min"),
            bench.get("median"),
            bench.get("mean"),
            bench.get("max"),
            bench.get("rounds"),
            trial_id,
        ],
    )


def list_trials(conn, *, active_only: bool = False) -> list[dict[str, Any]]:
    """All trials (or only ``running`` ones) as dicts, ordered by id."""
    sql = (
        "SELECT trial.trial_id, trial.parent_trial_id, trial.run_id, trial.goal, "
        "trial.hypothesis, CAST(trial.changed_files AS VARCHAR), trial.status, "
        "trial.verdict, CAST(trial.evidence_refs AS VARCHAR), trial.notes, "
        "COALESCE(trial.bench_mean_us, run.bench_mean_us), "
        "trial.bench_min_us, trial.bench_median_us, trial.bench_max_us, "
        "trial.bench_rounds FROM trial "
        "LEFT JOIN run ON run.run_id = trial.run_id"
    )
    if active_only:
        sql += " WHERE trial.status = 'running'"
    sql += " ORDER BY trial.trial_id"
    rows = conn.execute(sql).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "trial_id": row[0],
                "parent_trial_id": row[1],
                "run_id": row[2],
                "goal": row[3],
                "hypothesis": row[4],
                "changed_files": json.loads(row[5] or "[]"),
                "status": row[6],
                "verdict": row[7],
                "evidence_refs": json.loads(row[8] or "[]"),
                "notes": row[9],
                "bench_mean_us": row[10],
                "bench_min_us": row[11],
                "bench_median_us": row[12],
                "bench_max_us": row[13],
                "bench_rounds": row[14],
            }
        )
    return out
