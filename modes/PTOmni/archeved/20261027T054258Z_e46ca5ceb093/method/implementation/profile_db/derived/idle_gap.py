# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measured core holes and occupancy-derived drain tails.

Dispatch attribution comes from the official runtime counter timeline,
not a FIN-ready/not-started predicate over an idle hole.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Mapping, Sequence

from profile_db.derived.types import EdgeRec, GapWindow, RowSlice, TaskTiming
from profile_db.facts import Evidence

GAP_RECORD_US = 5.0

# idle_gap default identity: engine + core + gap bounds content; the
# surrogate id is assigned by the writer at insertion.
DRAIN_TAIL = "drain_tail"


@dataclass(frozen=True)
class GapRow:
    """One recorded core gap with its classification and payload."""

    engine: str
    core_index: int
    t0_us: float
    t1_us: float
    kind: str
    ready_task_ids: tuple | None
    evidence: str


def build_gaps(
    rows: Sequence[RowSlice],
    tasks: Mapping[str, TaskTiming],
    edges: Sequence[EdgeRec],
    windows: Mapping[str, GapWindow],
    level: int,
) -> tuple[GapRow, ...]:
    """Per-core gaps ≥ 5 µs with deterministic classification."""
    if not rows:
        return ()

    def classify(engine, t0_us, t1_us):
        window = windows.get(engine)
        if window is not None and window.drain_first is not None:
            band = bisect.bisect_right(window.band_ends, t0_us)
            if band >= window.drain_first:
                return DRAIN_TAIL, None, Evidence.PROVEN.value
        return "idle", None, Evidence.MEASURED.value

    by_core: dict[int, list[RowSlice]] = {}
    for row in rows:
        by_core.setdefault(row.core_index, []).append(row)

    out: list[GapRow] = []
    for core in sorted(by_core):
        core_rows = sorted(by_core[core], key=lambda r: (r.start_us, r.end_us, r.task_id))
        engine = core_rows[0].engine
        end = core_rows[0].end_us
        for nxt in core_rows[1:]:
            t0_us, t1_us = end, nxt.start_us
            end = max(end, nxt.end_us)
            if t1_us - t0_us < GAP_RECORD_US:
                continue
            kind, payload, evidence = classify(engine, t0_us, t1_us)
            out.append(
                GapRow(
                    engine=engine,
                    core_index=core,
                    t0_us=t0_us,
                    t1_us=t1_us,
                    kind=kind,
                    ready_task_ids=payload,
                    evidence=evidence,
                )
            )
    out.sort(key=lambda g: (g.engine, g.core_index, g.t0_us, g.t1_us))
    return tuple(out)


def overlap_us(window_t0: float, window_t1: float, start: float, end: float) -> float:
    """Clipped overlap of ``[start, end)`` with ``[window_t0, window_t1)``.

    Empty or inverted windows yield 0. The helper is shared by the idle-window
    occupancy query so the clip arithmetic lives in one tested place.
    """
    lo = max(window_t0, start)
    hi = min(window_t1, end)
    return max(0.0, hi - lo)
