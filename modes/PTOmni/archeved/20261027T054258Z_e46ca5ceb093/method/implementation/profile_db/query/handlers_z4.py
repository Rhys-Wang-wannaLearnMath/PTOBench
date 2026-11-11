# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Z4 handlers: why a task was late or long, its physical rows, the
scheduler phases around it, and the early-dispatch proof.

Timeline decomposition reuses the derived layer's tested pure functions
(stall / early_dispatch) — the same functions that produce the values the
T3 derivators store, so ``fin_detect + dispatch_wait + start_wait == gap``
holds here identically."""

from __future__ import annotations

from profile_db.derived import early_dispatch, stall
from profile_db.derived.types import RowSlice, TaskTiming
from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.registry import register
from profile_db.query.params import (
    CriticalPathParams,
    EarlyDispatchParams,
    MemoryParams,
    PerfHintsParams,
    PmuParams,
    RowsParams,
    SchedulerParams,
    WhyLateParams,
    WhyLongParams,
)

_ED_EVIDENCE = {
    early_dispatch.FULL: Evidence.PROVEN,
    early_dispatch.PARTIAL: Evidence.PROVEN,
    early_dispatch.NONE: Evidence.UNPROVEN,
    early_dispatch.UNAVAILABLE: Evidence.UNAVAILABLE,
}


def _upstream_depth(conn, run_id: int, task_id: str) -> int:
    """Longest producer chain above ``task_id`` (0 when it has none).

    ``UNION`` (not ``UNION ALL``) is load-bearing: the recursion walks a
    DAG, so ``UNION ALL`` enumerates every distinct *path* rather than
    every node. On a 266-task / 2546-edge capture that is ~11M
    materialized rows for one call; deduplicating on (task_id, depth)
    brings the same answer back in ~490.
    """
    row = common.one(
        conn,
        "WITH RECURSIVE up(task_id, depth) AS ("
        "  SELECT ?, 0 "
        "  UNION "
        "  SELECT e.pred, up.depth + 1 FROM dep_edge e "
        "  JOIN up ON e.run_id = ? AND e.succ = up.task_id "
        "  JOIN task p ON p.run_id = e.run_id AND p.task_id = e.pred"
        ") SELECT MAX(depth) FROM up",
        [task_id, run_id],
    )
    return int(row[0]) if row is not None and row[0] is not None else 0


def _rows_of(conn, run_id: int, task_id: str) -> list[RowSlice]:
    rows = common.q(
        conn,
        "SELECT core_index, engine, start_us, end_us, dispatch_us, receive_us, "
        "finish_us FROM task_row WHERE run_id = ? AND task_id = ? "
        "ORDER BY core_index, start_us",
        [run_id, task_id],
    )
    return [
        RowSlice(
            task_id=task_id,
            core_index=core,
            engine=engine or "",
            start_us=start,
            end_us=end,
            dispatch_us=dispatch,
            receive_us=receive,
            finish_us=finish,
        )
        for core, engine, start, end, dispatch, receive, finish in rows
    ]


def _timing_map(conn, run_id: int, pred_ids: list[str]) -> dict[str, TaskTiming]:
    if not pred_ids:
        return {}
    marks = ", ".join("?" for _ in pred_ids)
    rows = common.q(
        conn,
        f"SELECT task_id, max_finish_us, early_dispatch_flag, max_end_us FROM task "
        f"WHERE run_id = ? AND task_id IN ({marks})",
        [run_id, *pred_ids],
    )
    return {
        str(r[0]): TaskTiming(
            task_id=str(r[0]),
            engine=None,
            early_dispatch_flag=bool(r[2]),
            num_rows=0,
            busy_us=None,
            wall_us=None,
            min_dispatch_us=None,
            min_receive_us=None,
            min_start_us=None,
            max_end_us=r[3],
            max_finish_us=r[1],
        )
        for r in rows
    }


def _pred_ids(conn, run_id: int, task_id: str) -> list[str]:
    rows = common.q(
        conn,
        "SELECT pred FROM dep_edge WHERE run_id = ? AND succ = ? ORDER BY edge_id",
        [run_id, task_id],
    )
    deduped: list[str] = []
    for (pred,) in rows:
        if pred not in deduped:
            deduped.append(pred)
    return deduped


@register(
    "why_late",
    "Attribute: why could this task not start earlier — how long was each "
    "FIN -> dispatch -> receive -> start segment?",
    WhyLateParams,
)
def why_late(conn, params: WhyLateParams) -> list[Fact]:
    run_id = params.run_id
    task = common.task_row(conn, run_id, params.task_id)
    if task is None:
        return [
            Fact(
                "STALL",
                common.fields(run_id=run_id, task_id=params.task_id),
                Evidence.UNAVAILABLE,
            )
        ]
    level = int(common.one(conn, "SELECT swimlane_level FROM run WHERE run_id = ?", [run_id])[0])
    depth = _upstream_depth(conn, run_id, params.task_id)
    pred_ids = _pred_ids(conn, run_id, params.task_id)
    dec = stall.decompose_stall(
        _rows_of(conn, run_id, params.task_id), pred_ids, _timing_map(conn, run_id, pred_ids), level
    )
    if dec.gap_us is None:
        return [
            Fact(
                "STALL",
                common.fields(run_id=run_id, task_id=params.task_id, upstream_depth=depth),
                Evidence.UNAVAILABLE,
            )
        ]
    return [
        Fact(
            "STALL",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                data_ready_us=common.us(dec.data_ready_us),
                observed_ready_us=common.us(dec.ready_us),
                dispatch_us=common.us(dec.dispatch_us),
                receive_us=common.us(dec.receive_us),
                start_us=common.us(dec.start_us),
                fin_observation_us=common.us(dec.fin_observation_us),
                observed_to_dispatch_us=common.us(dec.fin_detect_us),
                post_dispatch_us=common.us(dec.dispatch_wait_us),
                local_setup_us=common.us(dec.start_wait_us),
                observed_ready_to_start_us=common.us(dec.gap_us),
                data_ready_to_start_us=common.us(dec.start_us - dec.data_ready_us),
                upstream_depth=depth,
            ),
            Evidence.MEASURED,
        )
    ]


@register(
    "why_long",
    "Attribute: why does this task run long, and where does it rank against "
    "its own family?",
    WhyLongParams,
)
def why_long(conn, params: WhyLongParams) -> list[Fact]:
    run_id = params.run_id
    task = common.task_row(conn, run_id, params.task_id)
    if task is None:
        return [
            Fact("LONG", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    family = task[2]
    busy_us = task[9]
    fields = common.fields(
        run_id=run_id,
        task_id=params.task_id,
        name=task[1],
        family=family,
        busy_us=common.us(busy_us),
        wall_us=common.us(task[10]),
        num_rows=task[8],
    )
    if family is not None:
        vals = [
            r[0]
            for r in common.q(
                conn,
                "SELECT busy_us FROM task WHERE run_id = ? AND family = ? "
                "AND busy_us IS NOT NULL ORDER BY busy_us",
                [run_id, family],
            )
        ]
        if vals:
            median = vals[(len(vals) - 1) // 2]
            fields["family_median_us"] = common.us(median)
            fields["family_rank"] = 1 + sum(1 for v in vals if v < busy_us)
            fields["family_tasks"] = len(vals)
    row_durations = [
        r[1] - r[0]
        for r in common.q(
            conn,
            "SELECT start_us, end_us FROM task_row WHERE run_id = ? AND task_id = ? "
            "ORDER BY core_index, start_us",
            [run_id, params.task_id],
        )
    ]
    if row_durations:
        fields["min_row_us"] = common.us(min(row_durations))
        fields["max_row_us"] = common.us(max(row_durations))
    return [Fact("LONG", fields, Evidence.MEASURED)]


@register(
    "rows",
    "Locate: what does this operator's physical row-level timing look like?",
    RowsParams,
)
def rows(conn, params: RowsParams) -> list[Fact]:
    run_id = params.run_id
    if common.task_row(conn, run_id, params.task_id) is None:
        return [
            Fact("ROW", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    rows = common.q(
        conn,
        "SELECT core_index, row_index, start_us, end_us, dispatch_us, receive_us, "
        "finish_us FROM task_row WHERE run_id = ? AND task_id = ? "
        "ORDER BY core_index, row_index",
        [run_id, params.task_id],
    )
    if not rows:
        return [
            Fact("ROW", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    return [
        Fact(
            "ROW",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                core_index=core,
                row_index=row_index,
                start_us=common.us(start),
                end_us=common.us(end),
                dispatch_us=common.us(dispatch),
                receive_us=common.us(receive),
                finish_us=common.us(finish),
            ),
            Evidence.MEASURED,
        )
        for core, row_index, start, end, dispatch, receive, finish in rows
    ]


@register(
    "scheduler",
    "Attribute: what did the scheduler/orchestrator phases do around this "
    "task?",
    SchedulerParams,
)
def scheduler(conn, params: SchedulerParams) -> list[Fact]:
    from profile_db.query import runtime

    if params.task_id is None:
        return runtime.scheduler(conn, params)
    run_id = params.run_id
    task = common.task_row(conn, run_id, params.task_id)
    if task is None:
        return [
            Fact("SCHED", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    dispatch = task[11] if task[11] is not None else task[13]
    finish = task[15]
    if dispatch is None or finish is None:
        return [
            Fact("SCHED", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    margin = 1.0
    lo, hi = dispatch - margin, finish + margin
    facts: list[Fact] = []
    sched_rows = common.q(
        conn,
        "SELECT lane, kind, task_id, t0_us, t1_us, tasks_processed, loop_iter FROM scheduler_phase "
        "WHERE run_id = ? AND t0_us < ? AND t1_us > ? ORDER BY lane, t0_us",
        [run_id, hi, lo],
    )
    for lane, kind, phase_task, t0, t1, processed, loop in sched_rows:
        facts.append(
            Fact(
                "SCHED",
                common.fields(
                    run_id=run_id,
                    lane=lane,
                    kind=kind,
                    task_id=phase_task,
                    t0_us=common.us(t0),
                    t1_us=common.us(t1),
                    tasks_processed=processed,
                    loop_iter=loop,
                ),
                Evidence.MEASURED,
            )
        )
    orch_rows = common.q(
        conn,
        "SELECT lane, submit_idx, task_id, t0_us, t1_us FROM orch_phase "
        "WHERE run_id = ? AND task_id = ? ORDER BY submit_idx",
        [run_id, params.task_id],
    )
    for lane, submit_idx, task_id, t0, t1 in orch_rows:
        facts.append(
            Fact(
                "ORCH",
                common.fields(
                    run_id=run_id, lane=lane, submit_idx=submit_idx, task_id=task_id,
                    t0_us=common.us(t0), t1_us=common.us(t1),
                ),
                Evidence.MEASURED,
            )
        )
    if not facts:
        return [
            Fact("SCHED", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    return [*runtime.scheduler(conn, params), *facts]


@register(
    "early_dispatch",
    "Attribute: was the early-dispatch eligibility actually used, and how far "
    "does the proof reach?",
    EarlyDispatchParams,
)
def early(conn, params: EarlyDispatchParams) -> list[Fact]:
    run_id = params.run_id
    if common.task_row(conn, run_id, params.task_id) is None:
        return [
            Fact("EARLY", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    freq = common.one(conn, "SELECT clock_freq_hz FROM run WHERE run_id = ?", [run_id])[0]
    level = int(common.one(conn, "SELECT swimlane_level FROM run WHERE run_id = ?", [run_id])[0])
    pred_ids = _pred_ids(conn, run_id, params.task_id)
    pred_edges = [
        (r[0], r[1])
        for r in common.q(
            conn,
            "SELECT source, pred FROM dep_edge WHERE run_id = ? AND succ = ? ORDER BY edge_id",
            [run_id, params.task_id],
        )
    ]
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, pred in pred_edges:
        if pred not in seen:
            seen.add(pred)
            deduped.append((source, pred))
    from profile_db.runtime_adapter.storage import read_snapshot

    eligibility_doc = read_snapshot(conn, run_id, "deps")
    eligible = (eligibility_doc.get("payload") or {}).get("eligibility", {}).get(params.task_id)
    proof = early_dispatch.prove(
        _rows_of(conn, run_id, params.task_id),
        deduped,
        _timing_map(conn, run_id, pred_ids),
        freq,
        level,
        eligible,
    )
    return [
        Fact(
            "EARLY",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                status=proof.status,
                structurally_eligible=eligible,
                observed_ready_us=common.us(proof.ready_us),
                tol_us=common.us(proof.tol_us),
                proven_blocks=proof.proven_blocks,
                total_blocks=proof.total_blocks,
            ),
            _ED_EVIDENCE[proof.status],
        )
    ]


@register(
    "pmu",
    "Constraints: which pipe is close to saturation, and what is the total "
    "cycle count?",
    PmuParams,
    default_budget_bytes=32768,
)
def pmu(conn, params: PmuParams) -> list[Fact]:
    run_id = params.run_id
    status = common.modality_status_fact(conn, run_id, "pmu")
    if (common.task_row(conn, run_id, params.task_id) is None
            and common.one(conn, "SELECT 1 FROM pmu_counter WHERE run_id=? AND task_id=? LIMIT 1", [run_id, params.task_id]) is None):
        facts = [
            Fact("PMU", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
        return [*([status] if status is not None else []), *facts]
    rows = common.q(
        conn,
        "SELECT counter, SUM(value), SUM(total_cycles), COUNT(*) FROM pmu_counter "
        "WHERE run_id = ? AND task_id = ? GROUP BY counter ORDER BY counter",
        [run_id, params.task_id],
    )
    if not rows:
        facts = [
            Fact("PMU", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
        return [*([status] if status is not None else []), *facts]
    sample_count = int(
        common.one(
            conn,
            "SELECT COUNT(DISTINCT sample_seq) FROM pmu_counter "
            "WHERE run_id = ? AND task_id = ?",
            [run_id, params.task_id],
        )[0]
    )
    facts: list[Fact] = [
        Fact(
            "PMU_SUMMARY",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                samples=sample_count,
                counters=len(rows),
                measurements=sum(int(row[3]) for row in rows),
            ),
            Evidence.MEASURED,
        )
    ]
    for counter, value, total, measurements in rows:
        fields = common.fields(
            run_id=run_id, task_id=params.task_id, counter=counter, value=value,
            total_cycles=total, samples=measurements,
        )
        if total:
            fields["ratio"] = round(value / total, 6)
        facts.append(Fact("PMU", fields, Evidence.MEASURED))
    # Without a total-cycles column there is no denominator, so the
    # occupancy ratio this query exists to report is genuinely absent —
    # say so instead of quietly omitting the field.
    if not any(row[2] for row in rows):
        facts.append(
            Fact(
                "EVIDENCE",
                common.fields(
                    run_id=run_id,
                    task_id=params.task_id,
                    metric="ratio",
                    reason="pmu.csv carries no total-cycles column",
                ),
                Evidence.UNAVAILABLE,
            )
        )
    if params.samples:
        samples = common.q(
            conn,
            "SELECT sample_seq, task_id_raw, thread_id, core_id, func_id, core_type, "
            "event_type, counter, value, total_cycles FROM pmu_counter "
            "WHERE run_id = ? AND task_id = ? "
            "ORDER BY sample_seq, counter",
            [run_id, params.task_id],
        )
        grouped: dict[int, dict] = {}
        for seq, raw, thread, core, func, core_type, event_type, counter, value, total in samples:
            item = grouped.setdefault(
                int(seq),
                {
                    "task_id_raw": raw,
                    "thread_id": thread,
                    "core_id": core,
                    "func_id": func,
                    "core_type": core_type,
                    "event_type": event_type,
                    "total_cycles": total,
                    "counters": {},
                },
            )
            item["counters"][counter] = value
        facts.extend(
            Fact(
                "PMU_SAMPLE",
                common.fields(
                    run_id=run_id,
                    task_id=params.task_id,
                    sample_seq=seq,
                    task_id_raw=item["task_id_raw"],
                    thread_id=item["thread_id"],
                    core_id=item["core_id"],
                    func_id=item["func_id"],
                    core_type=item["core_type"],
                    event_type=item["event_type"],
                    total_cycles=item["total_cycles"],
                    counters=item["counters"],
                ),
                Evidence.MEASURED,
            )
            for seq, item in grouped.items()
        )
    return [*([status] if status is not None else []), *facts]


@register(
    "critical_path",
    "Locate: which chain actually decides the duration (observed/static), and "
    "at which tasks does stall blow up?",
    CriticalPathParams,
    default_budget_bytes=32768,
)
def critical_path(conn, params: CriticalPathParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("PATH", run_id)
    rows = common.q(
        conn,
        "SELECT seq, task_id, wall_us, busy_us, compute_us, stall_us, gap_us, "
        "gap_kind, early_dispatch_proven FROM cpm_path "
        "WHERE run_id = ? AND kind = ? ORDER BY seq",
        [run_id, params.kind],
    )
    if not rows:
        return [
            Fact("PATH", common.fields(run_id=run_id, kind=params.kind), Evidence.UNAVAILABLE)
        ]
    identities = common.task_identities(
        conn, run_id, [task_id for _, task_id, *_ in rows if task_id is not None]
    )
    facts: list[Fact] = []
    for seq, task_id, wall, busy, compute, stall_us, gap, gap_kind, early_status in rows:
        name, family, engine = identities.get(str(task_id), (None, None, None))
        facts.append(
            Fact(
                "PATH",
                common.fields(
                    run_id=run_id,
                    kind=params.kind,
                    seq=seq,
                    task_id=task_id,
                    name=name,
                    family=family,
                    engine=engine,
                    wall_us=common.us(wall),
                    busy_us=common.us(busy),
                    compute_us=common.us(compute),
                    stall_us=common.us(stall_us),
                    gap_us=common.us(gap),
                    gap_kind=gap_kind,
                    early_dispatch_proven=early_status,
                ),
                Evidence.MEASURED,
            )
        )
    return facts


@register(
    "perf_hints",
    "Constraints: what tile/placement hints did the compiler emit?",
    PerfHintsParams,
)
def perf_hints(conn, params: PerfHintsParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("PERF_HINT", run_id)
    rows = common.q(
        conn,
        "SELECT seq, text, source_path, origin FROM perf_hint WHERE run_id = ? ORDER BY seq",
        [run_id],
    )
    if not rows:
        return [Fact("PERF_HINT", common.fields(run_id=run_id), Evidence.UNAVAILABLE)]
    return [
        Fact(
            "PERF_HINT",
            common.fields(run_id=run_id, seq=seq, text=text, source_path=source_path, origin=origin),
            Evidence.MEASURED,
        )
        for seq, text, source_path, origin in rows
    ]


@register(
    "memory",
    "Constraints: how far is each buffer space from its hardware limit?",
    MemoryParams,
)
def memory(conn, params: MemoryParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("MEMORY", run_id)
    rows = common.q(
        conn,
        "SELECT kernel, space, usage, limit_value FROM memory_entry "
        "WHERE run_id = ? ORDER BY kernel, space",
        [run_id],
    )
    if not rows:
        return [Fact("MEMORY", common.fields(run_id=run_id), Evidence.UNAVAILABLE)]
    ir = common.one(conn, "SELECT 1 FROM artifact WHERE run_id=? AND kind='allocation_ir'", [run_id])
    return [
        Fact(
            "MEMORY",
            common.fields(run_id=run_id, kernel=kernel, space=space, usage=usage, limit_value=limit_value,
                          headroom_bytes=limit_value - usage if limit_value is not None else None,
                          basis='allocated_address_extent' if ir else 'compiler_memory_report'),
            Evidence.PROVEN if ir else Evidence.MEASURED,
        )
        for kernel, space, usage, limit_value in rows
    ]
