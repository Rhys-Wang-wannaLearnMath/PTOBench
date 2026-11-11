# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared helpers for the query handlers.

The query layer reads only schema tables (and, for pure numeric reuse,
the derived layer's already-tested functions — never the ingest parsers
or the raw JSON artifacts). Every read is ordered so the emitted facts
are byte-for-byte deterministic.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from profile_db.errors import QueryError
from profile_db.facts import Evidence, Fact


def q(conn, sql: str, params: Iterable[Any] = ()) -> list[tuple]:
    """Run a read and return all rows."""
    return conn.execute(sql, list(params)).fetchall()


def one(conn, sql: str, params: Iterable[Any] = ()) -> tuple | None:
    return conn.execute(sql, list(params)).fetchone()


def num_key(value: str) -> tuple[int, int, str]:
    """Deterministic task-id ordering: numeric ids order numerically,
    anything else falls back to lexicographic order."""
    try:
        return (0, int(value), "")
    except ValueError:
        return (1, 0, value)


def us(value):
    """Display-round a microsecond value to nanosecond precision; the
    derived tables keep the raw floats — this only cleans the DSL output
    (avoids 1.7999999999999972-style noise for LLM readability)."""
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return value


def fields(**kwargs: Any) -> dict[str, Any]:
    """Drop ``None`` values so facts stay compact and readable."""
    return {key: value for key, value in kwargs.items() if value is not None}


def json_cell(text: str | None) -> Any:
    """Parse a stored JSON cell back into a Python structure; ``None`` and
    unparseable text pass through as ``None`` (never a fabricated value)."""
    if text is None or text == "":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def run_row(conn, run_id: int) -> tuple | None:
    return one(
        conn,
        "SELECT program, platform, device_id, CAST(captured_at AS VARCHAR), "
        "swimlane_level, clock_freq_hz, num_cores, CAST(core_types AS VARCHAR), "
        "rank_label, git_commit, git_dirty, CAST(runtime_cfg AS VARCHAR), "
        "bench_min_us, bench_median_us, bench_mean_us, bench_max_us, "
        "bench_rounds, makespan_us, raw_span_us, cpm_us, retained, "
        "CAST(notes AS VARCHAR), CAST(tags AS VARCHAR) FROM run WHERE run_id = ?",
        [run_id],
    )


def run_missing(rec: str, run_id: int) -> list[Fact]:
    """The canonical 'unavailable' answer for a run that does not exist."""
    return [Fact(rec, fields(run_id=run_id), Evidence.UNAVAILABLE)]


def modality_status_fact(conn, run_id: int, modality: str) -> Fact | None:
    """Return persisted optional-modality provenance when the run has it."""
    try:
        row = one(
            conn,
            "SELECT requested, request_value, rel_path, size_bytes, parser_state, "
            "entry_count, state, reason FROM modality_status "
            "WHERE run_id = ? AND modality = ?",
            [run_id, modality],
        )
    except Exception as exc:
        if "modality_status" in str(exc):
            return None
        raise
    if row is None:
        return None
    requested, request_value, rel_path, size_bytes, parser_state, entry_count, state, reason = row
    return Fact(
        "MODALITY",
        fields(
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


def task_row(conn, run_id: int, task_id: str) -> tuple | None:
    return one(
        conn,
        "SELECT task_id, name, family, engine, scope, early_dispatch_flag, "
        "CAST(kernel_ids AS VARCHAR), block_num, num_rows, busy_us, wall_us, "
        "min_dispatch_us, min_receive_us, min_start_us, max_end_us, "
        "max_finish_us, on_cpm_observed, on_cpm_static "
        "FROM task WHERE run_id = ? AND task_id = ?",
        [run_id, task_id],
    )


def task_identities(
    conn, run_id: int, task_ids: Sequence[str]
) -> dict[str, tuple[str | None, str | None, str | None]]:
    """task_id -> (name, family, engine) for the given ids, in one read.

    Ids absent from the result are not tasks of this run: ``dep_edge.pred``
    may name a host-side creator pseudo node (``source="creator"``), which
    the ingest layer preserves verbatim and which has no ``task`` row.
    Callers must treat a missing key as external, never as a task.
    """
    if not task_ids:
        return {}
    marks = ", ".join("?" for _ in task_ids)
    rows = q(
        conn,
        f"SELECT task_id, name, family, engine FROM task "
        f"WHERE run_id = ? AND task_id IN ({marks})",
        [run_id, *task_ids],
    )
    return {str(r[0]): (r[1], r[2], r[3]) for r in rows}


def engine_cores(conn, run_id: int) -> dict[str, int]:
    """Engine -> core count from the capture metadata (run.core_types)."""
    row = one(conn, "SELECT CAST(core_types AS VARCHAR) FROM run WHERE run_id = ?", [run_id])
    if row is None:
        return {}
    types = json_cell(row[0]) or []
    counts: dict[str, int] = {}
    for engine in types:
        counts[str(engine)] = counts.get(str(engine), 0) + 1
    return counts


def guard_window(t0_us: float, t1_us: float) -> None:
    if t1_us <= t0_us:
        raise QueryError(f"invalid window: t1_us={t1_us} must exceed t0_us={t0_us}")


def task_facts(conn, run_id: int, task_ids: Sequence[str]) -> list[Fact]:
    """TASK facts for many ids in one read, ordered like ``task_ids``.

    The single-id ``task_fact`` in a loop would issue one query per task,
    which a wide ``region`` window turns into hundreds of round trips.
    """
    if not task_ids:
        return []
    marks = ", ".join("?" for _ in task_ids)
    rows = q(
        conn,
        "SELECT task_id, name, family, engine, scope, early_dispatch_flag, "
        "CAST(kernel_ids AS VARCHAR), block_num, num_rows, busy_us, wall_us, "
        "min_dispatch_us, min_receive_us, min_start_us, max_end_us, "
        "max_finish_us, on_cpm_observed, on_cpm_static "
        f"FROM task WHERE run_id = ? AND task_id IN ({marks})",
        [run_id, *task_ids],
    )
    by_id = {str(row[0]): row for row in rows}
    return [
        _task_fact_from_row(run_id, by_id[task_id])
        for task_id in task_ids
        if task_id in by_id
    ]


def _task_fact_from_row(run_id: int, row: tuple) -> Fact:
    (
        task_id,
        name,
        family,
        engine,
        scope,
        early_flag,
        kernel_ids,
        block_num,
        num_rows,
        busy_us,
        wall_us,
        min_dispatch,
        min_receive,
        min_start,
        max_end,
        max_finish,
        on_observed,
        on_static,
    ) = row
    return Fact(
        "TASK",
        fields(
            run_id=run_id,
            task_id=task_id,
            name=name,
            family=family,
            engine=engine,
            scope=scope,
            early_dispatch_flag=early_flag,
            kernel_ids=json_cell(kernel_ids),
            block_num=block_num,
            num_rows=num_rows,
            busy_us=us(busy_us),
            wall_us=us(wall_us),
            min_dispatch_us=us(min_dispatch),
            min_receive_us=us(min_receive),
            min_start_us=us(min_start),
            max_end_us=us(max_end),
            max_finish_us=us(max_finish),
            on_cpm_observed=on_observed,
            on_cpm_static=on_static,
        ),
        Evidence.MEASURED,
    )


def task_fact(conn, run_id: int, task_id: str) -> Fact | None:
    """The canonical TASK fact for one logical task, or None when absent."""
    row = task_row(conn, run_id, task_id)
    if row is None:
        return None
    return _task_fact_from_row(run_id, row)


def gap_fact(run_id: int, row: tuple) -> Fact:
    """The canonical GAP fact from an idle_gap row
    (engine, core_index, t0_us, t1_us, kind, ready_task_ids, evidence)."""
    engine, core_index, t0, t1, kind, payload_text, evidence = row
    fact_fields = fields(
        run_id=run_id,
        engine=engine,
        core_index=core_index,
        t0_us=us(t0),
        t1_us=us(t1),
        kind=kind,
    )
    if kind not in ("idle", "drain_tail"):
        return Fact("GAP", fields(run_id=run_id, reason="legacy derived labels; refresh required"),
                    Evidence.UNAVAILABLE)
    return Fact("GAP", fact_fields, Evidence(evidence))


def dep_fact(run_id: int, row: tuple) -> Fact:
    """The canonical DEP fact from a dep_edge row ordered as
    (pred, succ, source, arg, flags, tensor_id, consumer_dtype,
    consumer_shape, consumer_start_offset, consumer_strides)."""
    (
        pred,
        succ,
        source,
        arg,
        flags,
        tensor_id,
        consumer_dtype,
        consumer_shape,
        consumer_start_offset,
        consumer_strides,
    ) = row
    return Fact(
        "DEP",
        fields(
            run_id=run_id,
            pred=pred,
            succ=succ,
            source=source,
            arg=arg,
            tensor_id=tensor_id,
            consumer_dtype=consumer_dtype,
            consumer_shape=json_cell(consumer_shape),
            consumer_start_offset=consumer_start_offset,
            consumer_strides=json_cell(consumer_strides),
            flags=json_cell(flags),
        ),
        Evidence.MEASURED,
    )


def chunk_bounds(n_bands: int, chunks: int) -> list[tuple[int, int]]:
    """Split ``n_bands`` stored bands into ``chunks`` contiguous, roughly
    equal buckets as (first_index, one_past_last_index)."""
    if chunks <= 0 or n_bands <= 0:
        return []
    size = -(-n_bands // chunks)  # ceil division
    out: list[tuple[int, int]] = []
    start = 0
    while start < n_bands:
        end = min(start + size, n_bands)
        out.append((start, end))
        start = end
    return out
