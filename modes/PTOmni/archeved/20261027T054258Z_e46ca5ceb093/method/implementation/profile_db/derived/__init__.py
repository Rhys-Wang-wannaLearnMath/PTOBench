# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Derive occupancy and timestamp facts; consume persisted runtime critical paths."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from profile_db.derived import cpm, early_dispatch, idle_gap, stall, time_band
from profile_db.derived.types import (
    DerivedResult,
    EdgeRec,
    RowSlice,
    TaskTiming,
    num_key,
)
from profile_db.errors import PfdbError


def derive_run(conn, run_id: int) -> DerivedResult:
    """Derive every T3 table row for one run from its stored tables."""
    run = conn.execute(
        "SELECT swimlane_level, clock_freq_hz, CAST(core_types AS VARCHAR) FROM run WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if run is None:
        raise PfdbError(f"run {run_id} does not exist")
    level = int(run[0] if run[0] is not None else 0)
    freq_hz = int(run[1]) if run[1] is not None else None
    if freq_hz is None:
        raise PfdbError(f"run {run_id} has no clock_freq_hz; cannot derive")
    try:
        core_types = json.loads(run[2] or "[]")
    except (TypeError, ValueError) as exc:
        raise PfdbError(f"run {run_id} core_types is corrupted: {exc}") from exc
    if not isinstance(core_types, list):
        core_types = []

    task_rows = conn.execute(
        "SELECT task_id, engine, early_dispatch_flag, num_rows, busy_us, wall_us, "
        "min_dispatch_us, min_receive_us, min_start_us, max_end_us, max_finish_us "
        "FROM task WHERE run_id = ? ORDER BY task_id",
        [run_id],
    ).fetchall()
    tasks: dict[str, TaskTiming] = {}
    for row in task_rows:
        timing = TaskTiming(
            task_id=str(row[0]),
            engine=str(row[1]) if row[1] is not None else None,
            early_dispatch_flag=bool(row[2]),
            num_rows=int(row[3] or 0),
            busy_us=float(row[4]) if row[4] is not None else None,
            wall_us=float(row[5]) if row[5] is not None else None,
            min_dispatch_us=float(row[6]) if row[6] is not None else None,
            min_receive_us=float(row[7]) if row[7] is not None else None,
            min_start_us=float(row[8]) if row[8] is not None else None,
            max_end_us=float(row[9]) if row[9] is not None else None,
            max_finish_us=float(row[10]) if row[10] is not None else None,
        )
        tasks[timing.task_id] = timing

    row_tuples = conn.execute(
        "SELECT task_id, core_index, engine, start_us, end_us, "
        "dispatch_us, receive_us, finish_us "
        "FROM task_row WHERE run_id = ? ORDER BY core_index, row_index, task_id",
        [run_id],
    ).fetchall()
    rows: list[RowSlice] = []
    for row in row_tuples:
        rows.append(
            RowSlice(
                task_id=str(row[0]),
                core_index=int(row[1]),
                engine=str(row[2]),
                start_us=float(row[3]) if row[3] is not None else 0.0,
                end_us=float(row[4]) if row[4] is not None else 0.0,
                dispatch_us=float(row[5]) if row[5] is not None else None,
                receive_us=float(row[6]) if row[6] is not None else None,
                finish_us=float(row[7]) if row[7] is not None else None,
            )
        )

    edge_tuples = conn.execute(
        "SELECT pred, succ, source FROM dep_edge WHERE run_id = ? ORDER BY edge_id",
        [run_id],
    ).fetchall()
    edges: list[EdgeRec] = [
        EdgeRec(pred=str(e[0]), succ=str(e[1]), source=str(e[2] or "")) for e in edge_tuples
    ]

    bands = time_band.build_bands(rows, core_types)
    windows = time_band.gap_windows(bands)
    gaps = idle_gap.build_gaps(rows, tasks, edges, windows, level)
    snapshot_row = conn.execute(
        "SELECT 1 FROM analysis_snapshot WHERE run_id=? AND family='critical_path'", [run_id]
    ).fetchone()
    facts = cpm.stored_paths(conn, run_id, rows) if snapshot_row else cpm.compute_paths(rows, edges, freq_hz)

    rows_by_task: dict[str, list[RowSlice]] = {}
    for row in rows:
        rows_by_task.setdefault(row.task_id, []).append(row)
    preds_by_succ: dict[str, list[str]] = {}
    pred_edges_by_succ: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        chain = preds_by_succ.setdefault(edge.succ, [])
        if edge.pred not in chain:
            chain.append(edge.pred)
            pred_edges_by_succ.setdefault(edge.succ, []).append((edge.source, edge.pred))

    observed_ids = [step.task_id for step in facts.observed]
    static_ids = [step.task_id for step in facts.static]

    from profile_db.runtime_adapter.storage import read_snapshot

    eligibility_doc = read_snapshot(conn, run_id, "deps")
    eligibility = (eligibility_doc.get("payload") or {}).get("eligibility", {})

    def early_status(task_id: str) -> str:
        return early_dispatch.classify(
            rows_by_task.get(task_id, ()),
            pred_edges_by_succ.get(task_id, ()),
            tasks,
            freq_hz,
            level,
            eligibility.get(task_id),
        )

    paths: list[dict[str, Any]] = []
    for step in facts.observed:
        task = tasks[step.task_id]
        gap_us = None
        if stall.has_fin_stream(level):
            ready = stall.ready_time_us(preds_by_succ.get(step.task_id, ()), tasks)
            if ready is not None:
                gap_us = step.start_us - ready
        paths.append(
            {
                "kind": "observed",
                "seq": step.seq,
                "task_id": step.task_id,
                "wall_us": task.wall_us,
                "busy_us": task.busy_us,
                "compute_us": step.compute_us,
                "stall_us": step.stall_us,
                "gap_us": gap_us,
                "gap_kind": step.kind,
                "early_dispatch_proven": early_status(step.task_id),
            }
        )
    for step in facts.static:
        task = tasks[step.task_id]
        paths.append(
            {
                "kind": "static",
                "seq": step.seq,
                "task_id": step.task_id,
                "wall_us": task.wall_us,
                "busy_us": task.busy_us,
                "compute_us": step.busy_us,
                "stall_us": None,
                "gap_us": None,
                "gap_kind": None,
                "early_dispatch_proven": early_status(step.task_id),
            }
        )

    return DerivedResult(
        bands=[_band_row(band) for band in bands],
        gaps=[_gap_row(gap) for gap in gaps],
        paths=paths,
        cpm_us=facts.cpm_us,
        task_flags=[
            {
                "task_id": task_id,
                "on_cpm_observed": task_id in observed_ids,
                "on_cpm_static": task_id in static_ids,
            }
            for task_id in sorted(tasks, key=num_key)
        ],
    )


def _band_row(band) -> dict[str, Any]:
    return {
        "band_idx": band.band_idx,
        "t0_us": band.t0_us,
        "t1_us": band.t1_us,
        "engine": band.engine,
        "total_cores": band.total_cores,
        "busy_cores": band.busy_cores,
        "task_ids": list(band.task_ids),
        "sparse": band.sparse,
        "drain_tail": band.drain_tail,
    }


def _gap_row(gap) -> dict[str, Any]:
    return asdict(gap)
