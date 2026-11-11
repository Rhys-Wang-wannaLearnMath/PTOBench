# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Z2 handlers: explain a time window or a single core.

``why_sparse`` reports the deterministic 6.3 classification of a density
bucket straight from the derived idle-gap rows; ``region`` enumerates the
activity and gaps inside a window; ``core`` walks one physical core."""

from __future__ import annotations

from typing import Any

from profile_db.errors import QueryError
from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.handlers_z1 import _load_bands
from profile_db.query.registry import register
from profile_db.query.params import CoreParams, IdleWindowParams, RegionParams, WhySparseParams


@register(
    "why_sparse",
    "Attribute: is this empty window a dispatch stall (tasks ready but not "
    "dispatched) or upstream starvation?",
    WhySparseParams,
)
def why_sparse(conn, params: WhySparseParams) -> list[Fact]:
    run_id = params.run_id
    if params.band is not None and params.stored_band is not None:
        raise QueryError("why_sparse takes either band (+bands) or stored_band, not both")
    if params.band is None and params.stored_band is None:
        raise QueryError(
            "why_sparse requires band (with the same bands value density used) "
            "or stored_band (as reported by sparse_regions)"
        )
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("SPARSE", run_id)
    by_engine = _load_bands(conn, run_id, params.engine)
    if not by_engine:
        return [
            Fact(
                "SPARSE",
                common.fields(
                    run_id=run_id, band=params.band, stored_band=params.stored_band
                ),
                Evidence.UNAVAILABLE,
            )
        ]
    reference = by_engine[sorted(by_engine)[0]]
    if params.stored_band is not None:
        window = [row for row in reference if row[0] == params.stored_band]
    else:
        # The bucket axis must be the one density produced; ``bands``
        # carries that so a non-default --bands cannot silently shift the
        # window this query explains.
        buckets = common.chunk_bounds(len(reference), params.bands)
        window = []
        if params.band < len(buckets):
            start, end = buckets[params.band]
            window = reference[start:end]
    if not window:
        return [
            Fact(
                "SPARSE",
                common.fields(
                    run_id=run_id,
                    band=params.band,
                    bands=params.bands if params.band is not None else None,
                    stored_band=params.stored_band,
                    engine=params.engine,
                ),
                Evidence.UNAVAILABLE,
            )
        ]
    t0, t1 = window[0][1], window[-1][2]

    from profile_db.query.runtime import counter_window

    return counter_window(conn, run_id, t0, t1, params.engine)


@register(
    "region",
    "Survey/attribute: what happened inside this time window, and why is this "
    "stretch empty?",
    RegionParams,
)
def region(conn, params: RegionParams) -> list[Fact]:
    common.guard_window(params.t0_us, params.t1_us)
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("REGION", run_id)

    row_sql = (
        "SELECT DISTINCT task_id FROM task_row WHERE run_id = ? "
        "AND start_us < ? AND end_us > ?"
    )
    row_args: list[Any] = [run_id, params.t1_us, params.t0_us]
    if params.core is not None:
        row_sql += " AND core_index = ?"
        row_args.append(params.core)
    if params.family is not None:
        row_sql += (
            " AND task_id IN (SELECT task_id FROM task WHERE run_id = ? AND family = ?)"
        )
        row_args.extend([run_id, params.family])
    row_sql += " ORDER BY task_id"
    task_ids = [r[0] for r in common.q(conn, row_sql, row_args)]

    gap_sql = (
        "SELECT engine, core_index, t0_us, t1_us, kind, CAST(ready_task_ids AS VARCHAR), "
        "evidence FROM idle_gap WHERE run_id = ? AND t0_us < ? AND t1_us > ?"
    )
    gap_args: list[Any] = [run_id, params.t1_us, params.t0_us]
    if params.core is not None:
        gap_sql += " AND core_index = ?"
        gap_args.append(params.core)
    gap_sql += " ORDER BY engine, core_index, t0_us"
    gaps = common.q(conn, gap_sql, gap_args)

    facts: list[Fact] = [
        Fact(
            "REGION",
            common.fields(
                run_id=run_id,
                t0_us=common.us(params.t0_us),
                t1_us=common.us(params.t1_us),
                family=params.family,
                core=params.core,
                tasks=len(task_ids),
                gaps=len(gaps),
            ),
            Evidence.MEASURED,
        )
    ]
    for task_id in task_ids:
        fact = common.task_fact(conn, run_id, task_id)
        if fact is None:
            continue
        facts.append(fact)
    facts.extend(common.gap_fact(run_id, gap) for gap in gaps)
    return facts


@register(
    "core",
    "Attribute: what were the other cores doing while this one sat idle?",
    CoreParams,
)
def core(conn, params: CoreParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("CORE", run_id)
    rows = common.q(
        conn,
        "SELECT task_id, row_index, start_us, end_us, dispatch_us, receive_us, "
        "finish_us FROM task_row WHERE run_id = ? AND core_index = ? "
        "ORDER BY start_us, end_us, task_id",
        [run_id, params.core],
    )
    engine = None
    if rows:
        engine = common.one(
            conn, "SELECT engine FROM task_row WHERE run_id = ? AND core_index = ? LIMIT 1",
            [run_id, params.core],
        )[0]
    else:
        types = common.json_cell(
            common.one(
                conn, "SELECT CAST(core_types AS VARCHAR) FROM run WHERE run_id = ?", [run_id]
            )[0]
        )
        if types and params.core < len(types):
            engine = str(types[params.core])
    gaps = common.q(
        conn,
        "SELECT engine, core_index, t0_us, t1_us, kind, CAST(ready_task_ids AS VARCHAR), "
        "evidence FROM idle_gap WHERE run_id = ? AND core_index = ? ORDER BY t0_us",
        [run_id, params.core],
    )
    facts: list[Fact] = [
        Fact(
            "CORE",
            common.fields(run_id=run_id, core_index=params.core, engine=engine, rows=len(rows), gaps=len(gaps)),
            Evidence.MEASURED,
        )
    ]
    for task_id, row_index, start, end, dispatch, receive, finish in rows:
        facts.append(
            Fact(
                "ROW",
                common.fields(
                    run_id=run_id,
                    core_index=params.core,
                    task_id=task_id,
                    row_index=row_index,
                    start_us=common.us(start),
                    end_us=common.us(end),
                    dispatch_us=common.us(dispatch),
                    receive_us=common.us(receive),
                    finish_us=common.us(finish),
                ),
                Evidence.MEASURED,
            )
        )
    facts.extend(common.gap_fact(run_id, gap) for gap in gaps)
    return facts


_OPPOSITE_ENGINE = {"aic": "aiv", "aiv": "aic"}


def _next_observed_successor(conn, run_id: int, task_id: str) -> str | None:
    """Next observed-CP task after ``task_id``, or None if it is not on
    that path or is the last hop."""
    row = common.one(
        conn,
        "SELECT seq FROM cpm_path WHERE run_id = ? AND kind = 'observed' "
        "AND task_id = ?",
        [run_id, task_id],
    )
    if row is None:
        return None
    nxt = common.one(
        conn,
        "SELECT task_id FROM cpm_path WHERE run_id = ? AND kind = 'observed' "
        "AND seq = ?",
        [run_id, int(row[0]) + 1],
    )
    return None if nxt is None else str(nxt[0])


@register(
    "idle_window",
    "Attribute: between this producer finishing and a later consumer starting, "
    "how occupied was the other engine, and which recorded gaps overlap "
    "that window?",
    IdleWindowParams,
    default_budget_bytes=16384,
)
def idle_window(conn, params: IdleWindowParams) -> list[Fact]:
    from profile_db.derived.idle_gap import overlap_us

    run_id = params.run_id
    after_id = params.after_task_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("WINDOW", run_id)
    after = common.task_row(conn, run_id, after_id)
    if after is None:
        return [
            Fact(
                "WINDOW",
                common.fields(run_id=run_id, after_task_id=after_id),
                Evidence.UNAVAILABLE,
            )
        ]
    until_id = params.until_task_id
    if until_id is None:
        on_path = common.one(
            conn,
            "SELECT 1 FROM cpm_path WHERE run_id = ? AND kind = 'observed' "
            "AND task_id = ?",
            [run_id, after_id],
        )
        if on_path is None:
            return [
                Fact(
                    "WINDOW",
                    common.fields(
                        run_id=run_id,
                        after_task_id=after_id,
                        reason="not-on-observed-cpm",
                    ),
                    Evidence.UNAVAILABLE,
                )
            ]
        until_id = _next_observed_successor(conn, run_id, after_id)
        if until_id is None:
            return [
                Fact(
                    "WINDOW",
                    common.fields(
                        run_id=run_id,
                        after_task_id=after_id,
                        reason="no-observed-successor",
                    ),
                    Evidence.UNAVAILABLE,
                )
            ]
    until = common.task_row(conn, run_id, until_id)
    if until is None:
        return [
            Fact(
                "WINDOW",
                common.fields(
                    run_id=run_id, after_task_id=after_id, until_task_id=until_id
                ),
                Evidence.UNAVAILABLE,
            )
        ]
    producer_engine = after[3]
    engine = params.engine
    if engine is None:
        engine = _OPPOSITE_ENGINE.get(str(producer_engine) if producer_engine else "")
        if engine is None:
            raise QueryError(
                "idle_window needs --engine when the producer engine is not "
                "aic or aiv"
            )
    t0 = after[14]
    t1 = until[13]
    if t0 is None or t1 is None:
        return [
            Fact(
                "WINDOW",
                common.fields(
                    run_id=run_id,
                    after_task_id=after_id,
                    until_task_id=until_id,
                    engine=engine,
                    reason="missing-timestamps",
                ),
                Evidence.UNAVAILABLE,
            )
        ]
    t0_us = float(t0)
    t1_us = float(t1)
    if t1_us <= t0_us:
        return [
            Fact(
                "WINDOW",
                common.fields(
                    run_id=run_id,
                    after_task_id=after_id,
                    until_task_id=until_id,
                    engine=engine,
                    t0_us=common.us(t0_us),
                    t1_us=common.us(t1_us),
                    reason="non-positive-window",
                ),
                Evidence.UNAVAILABLE,
            )
        ]
    window_us = t1_us - t0_us
    facts: list[Fact] = [
        Fact(
            "WINDOW",
            common.fields(
                run_id=run_id,
                after_task_id=after_id,
                until_task_id=until_id,
                engine=engine,
                t0_us=common.us(t0_us),
                t1_us=common.us(t1_us),
                window_us=common.us(window_us),
            ),
            Evidence.MEASURED,
        )
    ]
    cores = common.engine_cores(conn, run_id).get(engine, 0)
    row_hits = common.q(
        conn,
        "SELECT task_id, start_us, end_us FROM task_row "
        "WHERE run_id = ? AND engine = ? AND start_us < ? AND end_us > ? "
        "ORDER BY task_id, start_us",
        [run_id, engine, t1_us, t0_us],
    )
    busy_core_us = sum(
        overlap_us(t0_us, t1_us, float(start), float(end))
        for _, start, end in row_hits
        if start is not None and end is not None
    )
    capacity_us = float(cores) * window_us
    occupancy = (busy_core_us / capacity_us) if capacity_us else None
    facts.append(
        Fact(
            "OCCUPANCY",
            common.fields(
                run_id=run_id,
                engine=engine,
                cores=cores,
                busy_core_us=common.us(busy_core_us),
                capacity_us=common.us(capacity_us),
                occupancy=round(occupancy, 6) if occupancy is not None else None,
            ),
            Evidence.MEASURED if occupancy is not None else Evidence.UNAVAILABLE,
        )
    )
    gap_rows = common.q(
        conn,
        "SELECT kind, t0_us, t1_us FROM idle_gap "
        "WHERE run_id = ? AND engine = ? AND t0_us < ? AND t1_us > ? "
        "ORDER BY kind, t0_us",
        [run_id, engine, t1_us, t0_us],
    )
    by_kind: dict[str, list[float]] = {}
    for kind, gap_t0, gap_t1 in gap_rows:
        clipped = overlap_us(t0_us, t1_us, float(gap_t0), float(gap_t1))
        by_kind.setdefault(str(kind), []).append(clipped)
    for kind in sorted(by_kind):
        parts = by_kind[kind]
        facts.append(
            Fact(
                "GAP_SUMMARY",
                common.fields(
                    run_id=run_id,
                    engine=engine,
                    kind=kind,
                    count=len(parts),
                    total_us=common.us(sum(parts)),
                ),
                Evidence.MEASURED,
            )
        )
    task_ids = sorted({str(row[0]) for row in row_hits}, key=common.num_key)
    facts.extend(common.task_facts(conn, run_id, task_ids))
    return facts