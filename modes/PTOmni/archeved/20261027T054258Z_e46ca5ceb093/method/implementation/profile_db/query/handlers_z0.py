# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Z0 handlers: which runs exist, what one run is, where it came from.

These answer the orientation stage ("what is the baseline? which run is
the most recent usable one?") and the inventory stage ("which artifacts
does this run hold, in what configuration?")."""

from __future__ import annotations

from typing import Any

from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.registry import register
from profile_db.query.params import InventoryParams, OverviewParams, RunsListParams

_COUNT_SQL = {
    "tasks": "SELECT COUNT(*) FROM task WHERE run_id = ?",
    "rows": "SELECT COUNT(*) FROM task_row WHERE run_id = ?",
    "edges": "SELECT COUNT(*) FROM dep_edge WHERE run_id = ?",
    "artifacts": "SELECT COUNT(*) FROM artifact WHERE run_id = ?",
    "time_bands": "SELECT COUNT(*) FROM time_band WHERE run_id = ?",
    "idle_gaps": "SELECT COUNT(*) FROM idle_gap WHERE run_id = ?",
}


def _run_counts(conn, run_id: int) -> dict[str, int]:
    return {name: int(common.one(conn, sql, [run_id])[0]) for name, sql in _COUNT_SQL.items()}


@register(
    "runs_list",
    "Orient: what is the baseline, which platform/config, and which run is the latest usable one?",
    RunsListParams,
    rank_axis=True,
)
def runs_list(conn, params: RunsListParams) -> list[Fact]:
    # Listing is an orientation operation, not an aggregate. Returning every
    # run with its rank label is deterministic and lets callers choose a run id
    # before applying an optional rank guard to the follow-up query.
    sql = "SELECT run_id FROM run"
    args: list[Any] = []
    if params.rank is not None:
        sql += " WHERE rank_label = ?"
        args.append(params.rank)
    run_ids = [r[0] for r in common.q(conn, sql + " ORDER BY run_id", args)]
    facts: list[Fact] = []
    for run_id in run_ids:
        row = common.run_row(conn, run_id)
        counts = _run_counts(conn, run_id)
        trial = common.one(
            conn,
            "SELECT trial_id FROM trial WHERE run_id = ? ORDER BY trial_id DESC LIMIT 1",
            [run_id],
        )
        facts.append(
            Fact(
                "RUN",
                common.fields(
                    run_id=run_id,
                    program=row[0],
                    platform=row[1],
                    device_id=row[2],
                    level=row[4],
                    rank=row[8],
                    tasks=counts["tasks"],
                    rows=counts["rows"],
                    edges=counts["edges"],
                    makespan_us=common.us(row[17]),
                    bench_mean_us=common.us(row[14]),
                    retained=row[20],
                    trial_id=trial[0] if trial is not None else None,
                ),
                Evidence.MEASURED,
            )
        )
    return facts


@register(
    "overview",
    "Survey: what are this run's top-line metrics, topology, and graph size?",
    OverviewParams,
)
def overview(conn, params: OverviewParams) -> list[Fact]:
    run_id = params.run_id
    row = common.run_row(conn, run_id)
    if row is None:
        return common.run_missing("RUN", run_id)
    counts = _run_counts(conn, run_id)
    facts: list[Fact] = [
        Fact(
            "RUN",
            common.fields(
                run_id=run_id,
                program=row[0],
                platform=row[1],
                device_id=row[2],
                captured_at=row[3],
                level=row[4],
                clock_freq_hz=row[5],
                num_cores=row[6],
                rank=row[8],
                git_commit=row[9],
                git_dirty=row[10],
                retained=row[20],
            ),
            Evidence.MEASURED,
        ),
        Fact(
            "METRIC",
            common.fields(
                run_id=run_id,
                makespan_us=common.us(row[17]),
                cpm_us=common.us(row[19]),
                raw_span_us=common.us(row[18]),
                bench_min_us=common.us(row[12]),
                bench_median_us=common.us(row[13]),
                bench_mean_us=common.us(row[14]),
                bench_max_us=common.us(row[15]),
                bench_rounds=row[16],
                tasks=counts["tasks"],
                task_rows=counts["rows"],
                edges=counts["edges"],
                artifacts=counts["artifacts"],
                time_bands=counts["time_bands"],
                idle_gaps=counts["idle_gaps"],
            ),
            Evidence.MEASURED,
        ),
    ]
    cores_by_engine = common.engine_cores(conn, run_id)
    for engine in sorted(cores_by_engine):
        facts.append(
            Fact(
                "RESOURCE",
                common.fields(run_id=run_id, engine=engine, cores=cores_by_engine[engine]),
                Evidence.MEASURED,
            )
        )
    # A metric the capture level cannot supply is named explicitly rather
    # than just missing from METRIC: makespan needs the AICPU
    # dispatch/FIN stream, which level-1 captures do not carry.
    if row[17] is None:
        facts.append(
            Fact(
                "EVIDENCE",
                common.fields(
                    run_id=run_id,
                    metric="makespan_us",
                    reason="no-aicpu-fin-stream"
                    if row[4] is not None and int(row[4]) < 2
                    else "no-timed-rows",
                    level=row[4],
                ),
                Evidence.UNAVAILABLE,
            )
        )
    return facts


@register(
    "inventory",
    "Orient: which artifacts does this run hold, how are they stored, and can it be rebuilt?",
    InventoryParams,
)
def inventory(conn, params: InventoryParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("ARTIFACT", run_id)
    rows = common.q(
        conn,
        "SELECT kind, rel_path, sha256, size_bytes, store_mode FROM artifact "
        "WHERE run_id = ? ORDER BY kind, rel_path",
        [run_id],
    )
    facts = [
        Fact(
            "ARTIFACT",
            common.fields(
                run_id=run_id,
                kind=kind,
                rel_path=rel_path,
                sha256=sha256,
                size_bytes=size_bytes,
                store_mode=store_mode,
            ),
            Evidence.MEASURED,
        )
        for kind, rel_path, sha256, size_bytes, store_mode in rows
    ]
    statuses = common.q(
        conn,
        "SELECT modality, requested, request_value, rel_path, size_bytes, parser_state, "
        "entry_count, state, reason FROM modality_status WHERE run_id = ? ORDER BY modality",
        [run_id],
    )
    facts.extend(
        Fact(
            "MODALITY",
            common.fields(
                run_id=run_id,
                modality=modality,
                requested=requested,
                request_value=request_value,
                rel_path=rel_path,
                size_bytes=size_bytes,
                parser_state=parser_state,
                entry_count=entry_count,
                state=state,
                reason=reason,
            ),
            Evidence.MEASURED if state == "available" else Evidence.UNAVAILABLE,
        )
        for modality, requested, request_value, rel_path, size_bytes, parser_state, entry_count, state, reason in statuses
    )
    if params.analysis_details:
        from profile_db.runtime_adapter.storage import read_snapshot

        families = common.q(
            conn, "SELECT family FROM analysis_snapshot WHERE run_id=? ORDER BY family", [run_id]
        )
        for (family,) in families:
            doc = read_snapshot(conn, run_id, family)
            facts.append(
                Fact(
                    "ANALYSIS",
                    dict(run_id=run_id, **{k: v for k, v in doc.items() if k != "payload"}),
                    Evidence.PROVEN if doc["status"] == "ok" else Evidence.UNAVAILABLE,
                )
            )
    if not facts:
        return [Fact("ARTIFACT", common.fields(run_id=run_id), Evidence.UNAVAILABLE)]
    return facts
