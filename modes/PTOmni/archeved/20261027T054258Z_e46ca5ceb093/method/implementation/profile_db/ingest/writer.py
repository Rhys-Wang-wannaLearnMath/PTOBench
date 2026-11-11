# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run-table writers: plain SQL persistence, no domain logic.

All inserts are plain parameterized statements. JSON columns bind from
``json.dumps`` text with an explicit ``CAST(? AS JSON)``; list columns
bind native Python lists. Surrogate ids (run_id, edge_id, phase_id,
artifact_id) are assigned as ``max(id) + offset`` inside the caller's
write transaction, which also guarantees replace-mode consistency when a
capture is re-ingested: children are deleted first and re-inserted, so a
failed re-ingest rolls back to the previous complete state.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import duckdb

from profile_db.ingest.text_evidence import redact_paths
from profile_db.schema import RECORD_KINDS
from profile_db.task_ids import normalize_task_id


def next_id(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX({column}), 0) FROM {table}").fetchone()
    return int(row[0]) + 1


def find_run_by_records(conn: duckdb.DuckDBPyConnection, sha256: str, identity_key: str | None = None) -> int | None:
    placeholders = ", ".join("?" for _ in RECORD_KINDS)
    row = conn.execute(
        f"SELECT a.run_id FROM artifact a JOIN run r ON r.run_id=a.run_id WHERE kind IN ({placeholders}) AND sha256 = ? "
        "AND r.identity_key IS NOT DISTINCT FROM ? ORDER BY a.run_id LIMIT 1",
        [*RECORD_KINDS, sha256, identity_key],
    ).fetchone()
    return int(row[0]) if row else None


def delete_run_rows(conn: duckdb.DuckDBPyConnection, run_id: int, tables: Sequence[str]) -> None:
    for table in tables:
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])


def insert_run(conn: duckdb.DuckDBPyConnection, meta: Mapping[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO run (
            run_id, program, platform, device_id, captured_at, swimlane_level,
            clock_freq_hz, num_cores, core_types, core_to_thread, rank_label,
            git_commit, git_dirty, runtime_cfg, bench_min_us, bench_median_us,
            bench_mean_us, bench_max_us, bench_rounds, makespan_us, raw_span_us,
            cpm_us, retained, notes, tags
        ) VALUES (
            ?, ?, ?, ?, CAST(? AS TIMESTAMP), ?, ?, ?, CAST(? AS JSON),
            CAST(? AS JSON), ?, ?, ?, CAST(? AS JSON), ?, ?, ?, ?, ?,
            ?, ?, NULL, TRUE, ?, CAST(? AS VARCHAR[])
        )
        """,
        [
            meta["run_id"],
            meta["program"],
            meta.get("platform"),
            meta.get("device_id"),
            meta.get("captured_at"),
            meta["swimlane_level"],
            meta["clock_freq_hz"],
            meta["num_cores"],
            json.dumps(meta["core_types"]),
            json.dumps(meta["core_to_thread"]),
            meta.get("rank_label") or "single",
            meta.get("git_commit"),
            meta.get("git_dirty"),
            json.dumps(redact_paths(meta.get("runtime_cfg") or {})),
            meta.get("bench_min_us"),
            meta.get("bench_median_us"),
            meta.get("bench_mean_us"),
            meta.get("bench_max_us"),
            meta.get("bench_rounds"),
            meta.get("makespan_us"),
            meta.get("raw_span_us"),
            meta.get("notes"),
            json.dumps(list(meta.get("tags") or [])),
        ],
    )


def update_run(conn: duckdb.DuckDBPyConnection, run_id: int, meta: Mapping[str, Any]) -> None:
    conn.execute(
        """
        UPDATE run SET
            program = ?, platform = ?, device_id = ?, captured_at = CAST(? AS TIMESTAMP),
            swimlane_level = ?, clock_freq_hz = ?, num_cores = ?,
            core_types = CAST(? AS JSON), core_to_thread = CAST(? AS JSON),
            rank_label = ?,
            git_commit = ?, git_dirty = ?, runtime_cfg = CAST(? AS JSON),
            bench_min_us = ?, bench_median_us = ?, bench_mean_us = ?,
            bench_max_us = ?, bench_rounds = ?,
            makespan_us = ?, raw_span_us = ?, notes = ?, tags = CAST(? AS VARCHAR[])
        WHERE run_id = ?
        """,
        [
            meta["program"],
            meta.get("platform"),
            meta.get("device_id"),
            meta.get("captured_at"),
            meta["swimlane_level"],
            meta["clock_freq_hz"],
            meta["num_cores"],
            json.dumps(meta["core_types"]),
            json.dumps(meta["core_to_thread"]),
            meta.get("rank_label") or "single",
            meta.get("git_commit"),
            meta.get("git_dirty"),
            json.dumps(redact_paths(meta.get("runtime_cfg") or {})),
            meta.get("bench_min_us"),
            meta.get("bench_median_us"),
            meta.get("bench_mean_us"),
            meta.get("bench_max_us"),
            meta.get("bench_rounds"),
            meta.get("makespan_us"),
            meta.get("raw_span_us"),
            meta.get("notes"),
            json.dumps(list(meta.get("tags") or [])),
            run_id,
        ],
    )


def _executemany(conn, sql: str, rows: Sequence[tuple[Any, ...]]) -> None:
    """Parameterized batch insert; empty batches are legal no-ops (phase
    tables are legitimately empty on level-1 captures)."""
    if not rows:
        return
    prefix, marker, values = sql.partition('VALUES ')
    if sql.startswith('INSERT INTO ') and marker and values.startswith('(') and values.endswith(')'):
        import re
        match = re.fullmatch(r'INSERT INTO (\w+)\s*\(([^)]+)\)\s*',prefix)
        if match:
            # Binding each scalar makes DuckDB repeatedly probe optional
            # Python conversion packages. Bind one JSON string per batch and
            # cast in SQL using the actual table schema. No optional dataframe
            # dependency, SQL literals or per-value module probes are needed.
            table, names = match.groups()
            columns = [name.strip() for name in names.split(',')]
            types = {r[0]:r[1] for r in conn.execute('DESCRIBE '+table).fetchall()}
            numeric = {'TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT','UTINYINT','USMALLINT',
                       'UINTEGER','UBIGINT','UHUGEINT','FLOAT','DOUBLE','BOOLEAN'}
            projection = ','.join(f"CAST({'json_extract' if types[name] in numeric else 'json_extract_string'}"
                                  f"(value,'$[{i}]') AS {types[name]})" for i,name in enumerate(columns))
            for start in range(0,len(rows),128):
                batch = rows[start:start+128]
                conn.execute(prefix+'SELECT '+projection+' FROM json_each(?)',
                             [json.dumps(batch,allow_nan=False)])
            return
        conn.executemany(sql,rows)
    else:
        conn.executemany(sql, rows)


def insert_artifacts(
    conn: duckdb.DuckDBPyConnection, run_id: int, first_id: int, artifacts: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            artifact["kind"],
            artifact["rel_path"],
            artifact["sha256"],
            artifact["size_bytes"],
            artifact["store_mode"],
        )
        for index, artifact in enumerate(artifacts)
    ]
    _executemany(
        conn,
        "INSERT INTO artifact (artifact_id, run_id, kind, rel_path, sha256, size_bytes, store_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_tasks(conn: duckdb.DuckDBPyConnection, run_id: int, tasks: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        (
            run_id,
            task["task_id"],
            task.get("task_id_raw") or task["task_id"],
            normalize_task_id(task["task_id"]).u64,
            task["name"],
            task["family"],
            task["engine"],
            task["scope"],
            task["early_dispatch"],
            json.dumps(task["kernel_ids"]),
            task["block_num"],
            task["num_rows"],
            task["busy_us"],
            task["wall_us"],
            task["min_dispatch_us"],
            task["min_receive_us"],
            task["min_start_us"],
            task["max_end_us"],
            task["max_finish_us"],
        )
        for task in tasks
    ]
    _executemany(
        conn,
        "INSERT INTO task (run_id, task_id, task_id_raw, task_id_u64, name, family, engine, scope, "
        "early_dispatch_flag, kernel_ids, block_num, num_rows, busy_us, wall_us, "
        "min_dispatch_us, min_receive_us, min_start_us, max_end_us, max_finish_us) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_task_rows(conn: duckdb.DuckDBPyConnection, run_id: int, rows: Sequence[Mapping[str, Any]]) -> None:
    data = [
        (
            run_id,
            row["task_id"],
            row.get("task_id_raw") or row["task_id"],
            normalize_task_id(row["task_id"]).u64,
            row["core_id"],
            row["engine"],
            row["thread"],
            row["row_index"],
            row["start_us"],
            row["end_us"],
            row["dispatch_us"],
            row["receive_us"],
            row["finish_us"],
        )
        for row in rows
    ]
    _executemany(
        conn,
        "INSERT INTO task_row (run_id, task_id, task_id_raw, task_id_u64, core_index, engine, thread, "
        "row_index, start_us, end_us, dispatch_us, receive_us, finish_us) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        data,
    )


def insert_edges(
    conn: duckdb.DuckDBPyConnection, run_id: int, first_id: int, edges: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            edge["pred"],
            edge["succ"],
            normalize_task_id(edge["pred"]).u64,
            normalize_task_id(edge["succ"]).u64,
            edge.get("pred_raw") or edge["pred"],
            edge.get("succ_raw") or edge["succ"],
            edge["source"],
            edge["arg"],
            json.dumps(edge["flags"]),
            edge["tensor_id"],
            edge["consumer_dtype"],
            json.dumps(edge["consumer_shape"]),
            edge["consumer_start_offset"],
            json.dumps(edge["consumer_strides"]),
        )
        for index, edge in enumerate(edges)
    ]
    _executemany(
        conn,
        "INSERT INTO dep_edge (edge_id, run_id, pred, succ, pred_u64, succ_u64, pred_raw, succ_raw, source, arg, flags, "
        "tensor_id, consumer_dtype, consumer_shape, consumer_start_offset, consumer_strides) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?, CAST(? AS JSON), ?, CAST(? AS JSON))",
        rows,
    )


def insert_scheduler_phases(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    first_id: int,
    lanes: Sequence[Sequence[Mapping[str, Any]]],
) -> None:
    rows: list[tuple[Any, ...]] = []
    for lane, records in enumerate(lanes):
        for phase in records:
            # scheduler-schema records carry the acted-on task token on
            # dummy_task/predicated_skip/graph_prepare kinds; archived
            # lanes have none.
            raw_task = phase.get("task_id")
            phase_task = (
                normalize_task_id(int(raw_task)).canonical if raw_task is not None else None
            )
            rows.append(
                (
                    first_id + len(rows),
                    run_id,
                    lane,
                    phase.get("kind"),
                    phase_task,
                    phase.get("start_time_us"),
                    phase.get("end_time_us"),
                    phase.get("loop_iter"),
                    phase.get("tasks_processed"),
                    phase.get("pop_hit"),
                    phase.get("pop_miss"),
                    json.dumps(phase.get("shared_at_start", [])),
                    json.dumps(phase.get("shared_at_end", [])),
                )
            )
    _executemany(
        conn,
        "INSERT INTO scheduler_phase (phase_id, run_id, lane, kind, task_id, t0_us, t1_us, "
        "loop_iter, tasks_processed, pop_hit, pop_miss, shared_at_start, shared_at_end) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), CAST(? AS JSON))",
        rows,
    )


def insert_orchestrator_phases(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    lanes: Sequence[Sequence[Mapping[str, Any]]],
) -> None:
    rows: list[tuple[Any, ...]] = []
    for lane, records in enumerate(lanes):
        for phase in records:
            rows.append(
                (
                    run_id,
                    lane,
                    phase.get("submit_idx"),
                    str(phase["task_id"]) if phase.get("task_id") is not None else None,
                    phase.get("start_time_us"),
                    phase.get("end_time_us"),
                )
            )
    _executemany(
        conn,
        "INSERT INTO orch_phase (run_id, lane, submit_idx, task_id, t0_us, t1_us) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_perf_hints(
    conn: duckdb.DuckDBPyConnection, run_id: int, hints: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            hint["seq"],
            hint["text"],
            hint["source_path"],
            hint["origin"],
        )
        for hint in hints
    ]
    _executemany(
        conn,
        "INSERT INTO perf_hint (run_id, seq, text, source_path, origin) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def insert_memory_entries(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    first_id: int,
    entries: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            entry["kernel"],
            entry["space"],
            entry["usage"],
            entry["limit_value"],
        )
        for index, entry in enumerate(entries)
    ]
    _executemany(
        conn,
        "INSERT INTO memory_entry (memory_id, run_id, kernel, space, usage, limit_value) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_pmu_counters(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    first_id: int,
    counters: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            counter["task_id"],
            counter.get("task_id_raw") or counter["task_id"],
            counter.get("task_id_u64") or normalize_task_id(counter["task_id"]).u64,
            # Captures without an explicit sample coordinate are one legacy
            # aggregate sample; keep all counters in that sample bucket.
            counter.get("sample_seq", 0),
            counter.get("thread_id"),
            counter.get("core_id"),
            counter.get("func_id"),
            counter.get("core_type"),
            counter.get("event_type"),
            counter["counter"],
            counter["value"],
            counter.get("total_cycles"),
        )
        for index, counter in enumerate(counters)
    ]
    _executemany(
        conn,
        "INSERT INTO pmu_counter (pmu_id, run_id, task_id, task_id_raw, task_id_u64, "
        "sample_seq, thread_id, core_id, func_id, core_type, event_type, counter, value, total_cycles) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_modality_statuses(
    conn: duckdb.DuckDBPyConnection, run_id: int, statuses: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            item["modality"],
            item.get("requested"),
            item.get("request_value"),
            item.get("rel_path"),
            item.get("size_bytes"),
            item["parser_state"],
            item.get("entry_count"),
            item["state"],
            item.get("reason"),
        )
        for item in statuses
    ]
    _executemany(
        conn,
        "INSERT INTO modality_status (run_id, modality, requested, request_value, rel_path, "
        "size_bytes, parser_state, entry_count, state, reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_bench_strata(
    conn: duckdb.DuckDBPyConnection, run_id: int, strata: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            item["stratum"],
            item.get("source_sha256"),
            item.get("rounds"),
            item.get("warmup"),
            item.get("rank_count"),
            item.get("aggregation_mode"),
        )
        for item in strata
    ]
    _executemany(
        conn,
        "INSERT INTO bench_stratum (run_id, stratum, source_sha256, rounds, warmup, rank_count, aggregation_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_bench_samples(
    conn: duckdb.DuckDBPyConnection, run_id: int, samples: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (run_id, item["stratum"], item["round"], item["effective_us"])
        for item in samples
    ]
    _executemany(
        conn,
        "INSERT INTO bench_sample (run_id, stratum, round, effective_us) VALUES (?, ?, ?, ?)",
        rows,
    )


# ---------------------------------------------------------------------------
# Derived tables (T3): the derived layer computes the row lists, these
# writers only persist them.
# ---------------------------------------------------------------------------


def insert_time_bands(
    conn: duckdb.DuckDBPyConnection, run_id: int, bands: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            band["band_idx"],
            band["t0_us"],
            band["t1_us"],
            band["engine"],
            band["total_cores"],
            band["busy_cores"],
            json.dumps(band["task_ids"]),
            band["sparse"],
            band["drain_tail"],
        )
        for band in bands
    ]
    _executemany(
        conn,
        "INSERT INTO time_band (run_id, band_idx, t0_us, t1_us, engine, "
        "total_cores, busy_cores, task_ids, sparse, drain_tail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?)",
        rows,
    )


def insert_idle_gaps(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    first_id: int,
    gaps: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            gap["engine"],
            gap["core_index"],
            gap["t0_us"],
            gap["t1_us"],
            gap["kind"],
            json.dumps(gap["ready_task_ids"]) if gap["ready_task_ids"] is not None else None,
            gap["evidence"],
        )
        for index, gap in enumerate(gaps)
    ]
    _executemany(
        conn,
        "INSERT INTO idle_gap (gap_id, run_id, engine, core_index, t0_us, t1_us, "
        "kind, ready_task_ids, evidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?)",
        rows,
    )


def insert_cpm_paths(
    conn: duckdb.DuckDBPyConnection, run_id: int, paths: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            path["kind"],
            path["seq"],
            path["task_id"],
            path["wall_us"],
            path["busy_us"],
            path["compute_us"],
            path["stall_us"],
            path["gap_us"],
            path["gap_kind"],
            path["early_dispatch_proven"],
        )
        for path in paths
    ]
    _executemany(
        conn,
        "INSERT INTO cpm_path (run_id, kind, seq, task_id, wall_us, busy_us, "
        "compute_us, stall_us, gap_us, gap_kind, early_dispatch_proven) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def update_task_path_flags(
    conn: duckdb.DuckDBPyConnection, run_id: int, flags: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            flag["on_cpm_observed"],
            flag["on_cpm_static"],
            run_id,
            flag["task_id"],
        )
        for flag in flags
    ]
    _executemany(
        conn,
        "UPDATE task SET on_cpm_observed = ?, on_cpm_static = ? "
        "WHERE run_id = ? AND task_id = ?",
        rows,
    )


def update_run_cpm(conn: duckdb.DuckDBPyConnection, run_id: int, cpm_us: float | None) -> None:
    conn.execute("UPDATE run SET cpm_us = ? WHERE run_id = ?", [cpm_us, run_id])


# ---------------------------------------------------------------------------
# Extended-modality tables (T9): metadata-only inserts.
# ---------------------------------------------------------------------------


def insert_args_dump_entries(
    conn: duckdb.DuckDBPyConnection, run_id: int, entries: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            entry["seq"],
            entry["task_id"],
            entry.get("task_id_raw") or entry.get("task_id"),
            entry.get("task_id_u64") or normalize_task_id(entry.get("task_id")).u64
            if entry.get("task_id") is not None
            else None,
            entry["stage"],
            entry["role"],
            entry["arg_index"],
            entry["kind"],
            entry["dtype"],
            json.dumps(entry["shape"]),
            entry["bin_size"],
        )
        for entry in entries
    ]
    _executemany(
        conn,
        "INSERT INTO args_dump_entry (run_id, seq, task_id, task_id_raw, task_id_u64, stage, role, arg_index, "
        "kind, dtype, shape, bin_size) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?)",
        rows,
    )


def insert_scope_stats_entries(
    conn: duckdb.DuckDBPyConnection, run_id: int, entries: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            entry["seq"],
            entry["site"],
            entry["ring"],
            entry["phase"],
            json.dumps(entry["payload"]),
        )
        for entry in entries
    ]
    _executemany(
        conn,
        "INSERT INTO scope_stats_entry (run_id, seq, site, ring, phase, payload) "
        "VALUES (?, ?, ?, ?, ?, CAST(? AS JSON))",
        rows,
    )


def insert_incore_entries(
    conn: duckdb.DuckDBPyConnection, run_id: int, entries: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            entry["kernel"],
            entry["status"],
            entry["export_dir"],
            json.dumps(entry["metrics"]),
        )
        for entry in entries
    ]
    _executemany(
        conn,
        "INSERT INTO incore_entry (run_id, kernel, status, export_dir, metrics) "
        "VALUES (?, ?, ?, ?, CAST(? AS JSON))",
        rows,
    )
