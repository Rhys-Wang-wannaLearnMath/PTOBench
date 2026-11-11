# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cycle -> us conversion mirroring swimlane_converter semantics.

The AICore<->AICPU cross-domain *join* always goes through
``simpler_setup.tools.swimlane_converter.read_perf_data``. This module
replicates only the cycle origin (``base_time_cycles``), the host-mode
time-base splice, and the us conversion documented in that tool, so raw
AICore rows and phase records can be stored in the same time base as the
joined task rows. Parity tests pin these formulas against the converter
at zero tolerance.

Two record schemas are accepted (simpler unified the scheduler streams in
its 2026-09 "scheduler schema" change):

- current: ``scheduler_tasks{producer, records}`` and
  ``scheduler_records{streams[]}`` (metrics separated, optional run_epoch);
- archived: top-level ``aicpu_tasks`` and ``aicpu_scheduler_phases``.

Host-orchestrated captures (``metadata.orchestrator_source == "host"``)
carry ``host_orchestrator_phases`` stamped in host CLOCK_MONOTONIC ns.
The converter splices the two clock domains causally: every device-domain
us value gains ``host_composite_end_us``, and host phases are expressed
relative to the host origin. Both rules are mirrored here.
"""

from __future__ import annotations

from typing import Any

# Lifecycle cycle fields the converter's base_time tracking covers
# (mirrored field-for-field from simpler_setup.tools.swimlane_converter).
LIFECYCLE_CYCLE_FIELDS = (
    "handshake_start_cycles",
    "handshake_complete_cycles",
    "config_start_cycles",
    "topology_complete_cycles",
    "context_publish_start_cycles",
    "context_publish_complete_cycles",
    "bootstrap_wait_start_cycles",
    "bootstrap_complete_cycles",
    "register_release_start_cycles",
    "register_release_end_cycles",
    "exit_signal_start_cycles",
    "exit_signal_end_cycles",
    "exit_wait_start_cycles",
    "exit_wait_end_cycles",
)


def scheduler_task_rows(records: dict[str, Any]) -> list[Any]:
    """Scheduler dispatch/finish timing rows in either schema: current
    ``scheduler_tasks.records`` or the archived top-level
    ``aicpu_tasks``. Rows stay positional
    ``[core_id, reg_task_id, dispatch_cycles, finish_cycles, run_epoch?]``."""
    section = records.get("scheduler_tasks")
    if isinstance(section, dict):
        return list(section.get("records") or [])
    return list(records.get("aicpu_tasks") or [])


def scheduler_phase_record_lanes(records: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Scheduler phase record lanes in either schema: current
    ``scheduler_records.streams[].records`` (metrics are merged later by
    the swimlane loader) or the archived flat
    ``aicpu_scheduler_phases`` lanes."""
    section = records.get("scheduler_records")
    if isinstance(section, dict):
        return [list(stream.get("records") or []) for stream in section.get("streams") or []]
    return [list(lane) for lane in records.get("aicpu_scheduler_phases") or []]


def host_mode(records: dict[str, Any]) -> bool:
    """True when the capture is host-orchestrated (mirrors the
    converter's detection: explicit metadata, host phases, or a host
    capture block)."""
    metadata = records.get("metadata") or {}
    return (
        metadata.get("orchestrator_source") == "host"
        or bool(records.get("host_orchestrator_phases"))
        or isinstance(metadata.get("host_capture"), dict)
    )


def host_origin_ns(records: dict[str, Any]) -> int:
    """Host timeline origin: the metadata-declared origin, falling back
    to the earliest host phase timestamp (the converter's rule)."""
    metadata = records.get("metadata") or {}
    origin = int(metadata.get("host_orchestration_origin_ns") or metadata.get("host_timeline_origin_ns") or 0)
    timestamps = _host_timestamps(records)
    if timestamps and origin == 0:
        origin = min(timestamps)
    return origin


def host_composite_end_us(records: dict[str, Any]) -> int | float:
    """Length of the host-phase composite on the host timeline; the
    converter adds it to every device-domain us value so the device
    timeline starts where the host composite ends (causal splice)."""
    timestamps = _host_timestamps(records)
    if not timestamps:
        return 0.0
    return (max(timestamps) - host_origin_ns(records)) / 1000.0


def _host_timestamps(records: dict[str, Any]) -> list[int]:
    return [
        int(phase[field])
        for lane in records.get("host_orchestrator_phases") or []
        for phase in lane
        for field in ("start_host_ns", "end_host_ns")
    ]


def base_time_cycles(records: dict[str, Any]) -> int:
    """Minimum non-zero timestamp across every device stream (aicore rows,
    scheduler task rows, scheduler phases, orchestrator phases, aicpu
    lifecycle records). Mirrors the converter's base_time tracking; 0
    when no stream carries a timestamp."""
    base: int | None = None

    def track(value: Any) -> None:
        nonlocal base
        v = int(value)
        if v > 0 and (base is None or v < base):
            base = v

    for row in records.get("aicore_tasks") or []:
        start_c = int(row[3])
        r2s_c = int(row[5]) if len(row) > 5 else 0
        track(start_c - r2s_c)
        track(row[4])
    for row in scheduler_task_rows(records):
        track(row[2])
        track(row[3])
    for lane in scheduler_phase_record_lanes(records):
        for phase in lane:
            track(phase.get("start_cycles", 0))
            track(phase.get("end_cycles", 0))
    for lane in records.get("aicpu_orchestrator_phases") or []:
        for phase in lane:
            track(phase.get("start_cycles", 0))
            track(phase.get("end_cycles", 0))
    for record in records.get("aicpu_lifecycle_records") or []:
        for field in LIFECYCLE_CYCLE_FIELDS:
            track(record.get(field, 0))
    return base if base is not None else 0


def to_us(cycles: int, base: int, clock_freq_hz: int, host_offset_us: float = 0.0) -> float:
    """Cycles since the origin -> us; non-positive cycles map to 0.0 (the
    converter's own convention for synthesized/absent timestamps). On
    host-orchestrated captures every device-domain value is shifted by
    ``host_offset_us`` (the causal splice), which non-positive cycles
    stay exempt from. The multiplication groups as
    ``(cycles - base) * (1e6 / freq)`` exactly like the converter — the
    other grouping rounds differently in the last ulp and breaks the
    zero-tolerance parity."""
    if cycles <= 0:
        return 0.0
    return host_offset_us + (cycles - base) * (1_000_000.0 / float(clock_freq_hz))


def aicore_row_us(
    row: list[int], base: int, clock_freq_hz: int, host_offset_us: float = 0.0
) -> dict[str, Any]:
    """Convert one raw aicore row to its us-domain fields.

    Row layout: [core_id, task_token_raw, reg_task_id, start_cycles,
    end_cycles, receive_to_start_cycles?, run_epoch?]. Archived rows carry
    five or six columns; identity-stamped rows carry seven.
    """
    start_c = int(row[3])
    r2s = int(row[5]) if len(row) > 5 else 0
    return {
        "core_id": int(row[0]),
        "task_id": int(row[1]),
        "reg_task_id": int(row[2]),
        "start_us": to_us(start_c, base, clock_freq_hz, host_offset_us),
        "end_us": to_us(int(row[4]), base, clock_freq_hz, host_offset_us),
        "receive_us": to_us(start_c - r2s, base, clock_freq_hz, host_offset_us),
        "r2s_cycles": r2s,
    }


def host_phase_us(phase: dict[str, Any], origin_ns: int) -> dict[str, Any]:
    """Convert one host-orchestrator phase record (``start_host_ns`` /
    ``end_host_ns``) to us on the host timeline. A timestamp before the
    origin is unrenderable and rejected, mirroring the converter."""
    start_ns = int(phase.get("start_host_ns", 0))
    end_ns = int(phase.get("end_host_ns", 0))
    if start_ns < origin_ns or end_ns < start_ns:
        raise ValueError(
            f"invalid host orchestrator phase: origin={origin_ns}, start={start_ns}, end={end_ns}"
        )
    out = dict(phase)
    out["start_time_us"] = (start_ns - origin_ns) / 1000.0
    out["end_time_us"] = (end_ns - origin_ns) / 1000.0
    out.pop("start_host_ns", None)
    out.pop("end_host_ns", None)
    return out


def phase_us(
    phase: dict[str, Any], base: int, clock_freq_hz: int, host_offset_us: float = 0.0
) -> dict[str, Any]:
    """Convert one scheduler/orchestrator phase record to us-domain fields,
    keeping every non-timestamp key verbatim (the converter strips only
    the *_cycles columns)."""
    out = dict(phase)
    out["start_time_us"] = to_us(int(phase.get("start_cycles", 0)), base, clock_freq_hz, host_offset_us)
    out["end_time_us"] = to_us(int(phase.get("end_cycles", 0)), base, clock_freq_hz, host_offset_us)
    out.pop("start_cycles", None)
    out.pop("end_cycles", None)
    return out
