# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Z1 handlers: where the run is dense and where it is sparse.

These read the derived density index directly (no raw JSON, no re-parsing);
display-time aggregation into ``--bands`` buckets is deterministic and the
bucket index becomes the navigation coordinate for ``why_sparse``."""

from __future__ import annotations

from typing import Any

from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.registry import register
from profile_db.query.params import DensityParams, SparseRegionsParams


def _load_bands(conn, run_id: int, engine: str | None) -> dict[str, list[tuple]]:
    sql = (
        "SELECT band_idx, t0_us, t1_us, engine, total_cores, busy_cores, "
        "CAST(task_ids AS VARCHAR), sparse, drain_tail FROM time_band "
        "WHERE run_id = ?"
    )
    args: list[Any] = [run_id]
    if engine is not None:
        sql += " AND engine = ?"
        args.append(engine)
    sql += " ORDER BY engine, band_idx"
    by_engine: dict[str, list[tuple]] = {}
    for row in common.q(conn, sql, args):
        by_engine.setdefault(row[3], []).append(row)
    return by_engine


@register(
    "density",
    "Survey: where did the time go — which engine sits idle and which time "
    "bands collapse in occupancy?",
    DensityParams,
)
def density(conn, params: DensityParams) -> list[Fact]:
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [params.run_id]) is None:
        return common.run_missing("BAND", params.run_id)
    by_engine = _load_bands(conn, params.run_id, params.engine)
    if not by_engine:
        return [Fact("BAND", common.fields(run_id=params.run_id), Evidence.UNAVAILABLE)]
    facts: list[Fact] = []
    for engine in sorted(by_engine):
        stored = by_engine[engine]
        buckets = common.chunk_bounds(len(stored), params.bands)
        for bucket_idx, (start, end) in enumerate(buckets):
            chunk = stored[start:end]
            task_ids: list[str] = []
            for row in chunk:
                for task_id in common.json_cell(row[6]) or []:
                    if task_id not in task_ids:
                        task_ids.append(str(task_id))
            task_ids.sort(key=common.num_key)
            facts.append(
                Fact(
                    "BAND",
                    common.fields(
                        run_id=params.run_id,
                        band_idx=bucket_idx,
                        t0_us=common.us(chunk[0][1]),
                        t1_us=common.us(chunk[-1][2]),
                        engine=engine,
                        total_cores=chunk[0][4],
                        busy_cores=max(row[5] for row in chunk),
                        task_ids=task_ids,
                        sparse=all(row[7] for row in chunk),
                        drain_tail=any(row[8] for row in chunk),
                    ),
                    Evidence.MEASURED,
                )
            )
    return facts


@register(
    "sparse_regions",
    "Locate: which time bands collapse in occupancy, and what is each one "
    "blocked on?",
    SparseRegionsParams,
)
def sparse_regions(conn, params: SparseRegionsParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("SPARSE", run_id)
    sql = (
        "SELECT band_idx, t0_us, t1_us, engine, total_cores, busy_cores FROM time_band "
        "WHERE run_id = ? AND sparse"
    )
    args: list[Any] = [run_id]
    if params.engine is not None:
        sql += " AND engine = ?"
        args.append(params.engine)
    sql += " ORDER BY busy_cores ASC, band_idx ASC LIMIT ?"
    args.append(params.top_k)
    rows = common.q(conn, sql, args)
    facts: list[Fact] = []
    for band_idx, t0, t1, engine, total, busy in rows:
        evidence = Evidence.MEASURED
        fields = common.fields(
            run_id=run_id,
            # Storage band index (5 µs granularity), NOT a density display
            # bucket: feed why_sparse the t0_us/t1_us window below, never
            # this number as its ``band``.
            stored_band_idx=band_idx,
            t0_us=common.us(t0),
            t1_us=common.us(t1),
            engine=engine,
            total_cores=total,
            busy_cores=busy,
            kind="sparse",
        )
        facts.append(Fact("SPARSE", fields, evidence))
    if not facts:
        return [Fact("SPARSE", common.fields(run_id=run_id), Evidence.UNAVAILABLE)]
    return facts
