# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Swimlane loading: capture discovery, the AICore<->AICPU join, and
per-run aggregation.

The cross-domain join is **never** reimplemented here. For capture
levels 2-4 and calibrated level-1 captures the loader calls
``simpler_setup.tools.swimlane_converter.read_perf_data``, which owns the
join semantics (AICPU dispatch/finish mapped onto AICore rows), the
cycle origin (base_time), and the us conversion. Level-1 captures carry
no AICPU stream at all (empty by construction upstream), so they can take
a local conversion path that parity tests pin against the converter.

Two record schemas are accepted: the current scheduler schema
(``scheduler_tasks`` + ``scheduler_records.streams`` with per-record
metrics, merged by the converter on levels 2-4) and the archived flat keys
(``aicpu_tasks`` / ``aicpu_scheduler_phases``).
Host-orchestrated captures (``host_orchestrator_phases`` in host
CLOCK_MONOTONIC ns) are spliced onto the same causal composite the
converter builds. Legacy records files keyed ``l2_swimlane_level`` are
materialized to a temporary file carrying the ``chip_swimlane_level``
spelling the converter reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profile_db.errors import IngestError
from profile_db.ingest import swimlane_us
from profile_db.runtime_adapter.capture import (
    _merge_scheduler_streams as _merge_scheduler_streams,
    _scheduler_phase_lanes, read_capture, validate_capture,
)
from profile_db.task_ids import normalize_task_id


@dataclass
class RowTime:
    """One physical execution row, in us.

    Fields kept on one record so the writer stores the exact same numbers
    emitted by the converter (parity at zero tolerance), plus the raw
    per-core row ordinal recovered from the aicore stream.
    """

    task_id: str
    task_id_raw: str
    core_id: int
    row_index: int
    engine: str
    thread: int | None
    start_us: float
    end_us: float
    receive_us: float
    dispatch_us: float
    finish_us: float

    @property
    def busy_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def wall_us(self) -> float:
        return self.finish_us - self.dispatch_us


@dataclass
class Swimlane:
    """All swimlane facts for one run, ready for the writer."""

    level: int
    clock_freq_hz: int
    num_cores: int
    core_types: list[str]
    core_to_thread: list[int]
    makespan_us: float | None  # None on level 1: no AICPU dispatch/FIN stream
    raw_span_us: float
    rows: list[RowTime] = field(default_factory=list)
    scheduler_phases: list[list[dict[str, Any]]] = field(default_factory=list)
    orchestrator_phases: list[list[dict[str, Any]]] = field(default_factory=list)
    orchestrator_source: str = "aicpu"  # metadata-declared producer of the stream


def _orchestrator_phase_lanes(
    records: dict[str, Any], origin_ns: int
) -> list[list[dict[str, Any]]]:
    """Orchestrator lanes: device-cycled ``aicpu_orchestrator_phases`` or
    host-stamped ``host_orchestrator_phases`` (us on the host timeline,
    relative to the origin). Both present means an ambiguous clock-domain
    source and is rejected, mirroring the converter."""
    device_lanes = records.get("aicpu_orchestrator_phases") or []
    host_lanes = records.get("host_orchestrator_phases") or []
    if device_lanes and host_lanes:
        raise IngestError(
            "both AICPU and host orchestrator phases are present; clock-domain source is ambiguous"
        )
    if host_lanes:
        try:
            return [[swimlane_us.host_phase_us(phase, origin_ns) for phase in lane] for lane in host_lanes]
        except ValueError as exc:
            raise IngestError(str(exc)) from exc
    return [list(lane) for lane in device_lanes]


def _core_type_of(core_id: int, metadata: dict[str, Any]) -> str:
    core_types = metadata.get("core_types") or []
    if 0 <= core_id < len(core_types):
        return str(core_types[core_id])
    return "aiv"  # the converter's own out-of-range fallback


def _thread_of(core_id: int, metadata: dict[str, Any]) -> int | None:
    core_to_thread = metadata.get("core_to_thread") or []
    if 0 <= core_id < len(core_to_thread):
        return int(core_to_thread[core_id])
    return None


def _raw_row_ordinals(records: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    """(core_id, task_token) -> row ordinals in file order. The ordinals
    are the raw third column and are unique per core."""
    by_key: dict[tuple[int, int], list[int]] = {}
    for row in records.get("aicore_tasks") or []:
        by_key.setdefault((int(row[0]), int(row[1])), []).append(int(row[2]))
    return by_key


def _joined_rows_level1(
    records: dict[str, Any], base: int, freq: int, host_offset_us: float
) -> list[dict[str, Any]]:
    """Level-1 rows: no AICPU stream exists, so dispatch/finish stay 0.0
    (the converter synthesizes the same values)."""
    metadata = records.get("metadata") or {}
    rows = []
    for raw in records.get("aicore_tasks") or []:
        core_id, token = int(raw[0]), int(raw[1])
        start_c = int(raw[3])
        r2s = int(raw[5]) if len(raw) > 5 else 0
        rows.append(
            {
                "task_id": token,
                "core_id": core_id,
                "core_type": _core_type_of(core_id, metadata),
                "start_time_us": swimlane_us.to_us(start_c, base, freq, host_offset_us),
                "end_time_us": swimlane_us.to_us(int(raw[4]), base, freq, host_offset_us),
                "receive_time_us": swimlane_us.to_us(start_c - r2s, base, freq, host_offset_us),
                "dispatch_time_us": 0.0,
                "finish_time_us": 0.0,
            }
        )
    return rows


def load(records_path: Path, records: dict[str, Any]) -> Swimlane:
    """Load one records file into a Swimlane. Levels 2-4 and calibrated level 1 go through the
    upstream converter (the join); level 1 takes the local path."""
    level = int(records.get("chip_swimlane_level") or records.get("l2_swimlane_level"))
    if level not in (1, 2, 3, 4):
        raise IngestError(f"unsupported swimlane level: {level!r}")

    metadata = records.get("metadata") or {}
    clock_freq_hz = int(metadata.get("clock_freq_hz") or 0)
    if clock_freq_hz <= 0:
        raise IngestError("metadata is missing a positive clock_freq_hz")
    core_types = [str(t) for t in metadata.get("core_types") or []]
    core_to_thread = [int(t) for t in metadata.get("core_to_thread") or []]
    num_cores = int(metadata.get("num_cores") or len(core_types))
    if num_cores != len(core_types):
        raise IngestError(
            f"metadata num_cores={num_cores} does not match core_types length {len(core_types)}"
        )

    validate_capture(records)
    raw_rows = records.get("aicore_tasks") or []

    is_host_mode = swimlane_us.host_mode(records)
    origin_ns = swimlane_us.host_origin_ns(records)
    host_offset_us = swimlane_us.host_composite_end_us(records) if is_host_mode else 0.0
    base = swimlane_us.base_time_cycles(records)

    local_only = level == 1 and not isinstance(metadata.get("clock_anchors"), dict)
    if local_only:
        joined = {"tasks": _joined_rows_level1(records, base, clock_freq_hz, host_offset_us)}
    else:
        joined = read_capture(records_path, records)

    ordinals = _raw_row_ordinals(records)
    rows: list[RowTime] = []
    for task in joined["tasks"]:
        task_token = int(task["task_id"])
        core_id = int(task["core_id"])
        pending = ordinals.setdefault((core_id, task_token), [])
        if not pending:
            raise IngestError("upstream converter changed AICore task identity")
        row_index = pending.pop(0)
        task_identity = normalize_task_id(task_token)
        rows.append(
            RowTime(
                task_id=task_identity.canonical,
                task_id_raw=task_identity.raw,
                core_id=core_id,
                row_index=row_index,
                engine=_core_type_of(core_id, metadata),
                thread=_thread_of(core_id, metadata),
                start_us=float(task["start_time_us"]),
                end_us=float(task["end_time_us"]),
                receive_us=float(task.get("receive_time_us", 0.0)),
                dispatch_us=float(task.get("dispatch_time_us", 0.0)),
                finish_us=float(task.get("finish_time_us", 0.0)),
            )
        )

    if local_only:
        scheduler_phases = [
            [swimlane_us.phase_us(phase, base, clock_freq_hz, host_offset_us) for phase in lane]
            for lane in _scheduler_phase_lanes(records)
        ]
        orchestrator_phases = _orchestrator_phase_lanes(records, origin_ns)
    else:
        # Converter lanes are indexed by scheduler_id, including empty slots.
        scheduler_phases = [
            [dict(phase, kind=phase["phase"]) for phase in lane]
            for lane in joined.get("aicpu_scheduler_phases", [])
        ]
        orchestrator_phases = joined.get("aicpu_orchestrator_phases", [])

    # makespan = max(FIN) - min(dispatch) (DESIGN.md 5.3). Level-1 captures
    # carry no AICPU stream, so those two columns are 0.0 placeholders and
    # the difference would be a fabricated 0.0 µs run length; the design
    # forbids reading the placeholders as instants, so the column stays
    # NULL and the query layer reports it unavailable. raw_span_us still
    # holds — it only reads aicore start/end ticks.
    makespan = (
        max(r.finish_us for r in rows) - min(r.dispatch_us for r in rows)
        if rows and level >= 2
        else None
    )
    raw_span = (
        (max(int(r[4]) for r in raw_rows) - min(int(r[3]) for r in raw_rows))
        * (1_000_000.0 / float(clock_freq_hz))
        if raw_rows
        else 0.0
    )

    return Swimlane(
        level=level,
        clock_freq_hz=clock_freq_hz,
        num_cores=num_cores,
        core_types=core_types,
        core_to_thread=core_to_thread,
        makespan_us=makespan,
        raw_span_us=raw_span,
        rows=rows,
        scheduler_phases=scheduler_phases,
        orchestrator_phases=orchestrator_phases,
        orchestrator_source="host" if is_host_mode else "aicpu",
    )
