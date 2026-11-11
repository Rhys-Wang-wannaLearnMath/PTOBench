# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime structural eligibility and separate two-tick timestamp proof."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from profile_db.derived.types import RowSlice, TaskTiming
from profile_db.derived.stall import has_fin_stream

FULL = "full"
PARTIAL = "partial"
NONE = "none"
UNAVAILABLE = "unavailable"

TOL_TICKS = 2


@dataclass(frozen=True)
class EarlyProof:
    """The early-dispatch outcome plus the numbers behind it."""

    status: str
    ready_us: float | None
    tol_us: float | None
    proven_blocks: int | None
    total_blocks: int | None

    @staticmethod
    def unavailable() -> "EarlyProof":
        return EarlyProof(UNAVAILABLE, None, None, None, None)


def prove(
    succ_rows: Sequence[RowSlice],
    pred_edges: Sequence[tuple[str, str]],  # (edge source, pred task id), deduped
    tasks: Mapping[str, TaskTiming],
    freq_hz: int | None,
    level: int,
    eligible: bool | None = None,
) -> EarlyProof:
    """Prove early dispatch for one consumer, returning status + evidence."""
    if not succ_rows or not has_fin_stream(level) or freq_hz is None:
        return EarlyProof.unavailable()

    if eligible is None:
        from profile_db.runtime_adapter import module

        nodes = set(tasks) | {p for _, p in pred_edges}
        table = {t: dict(early_dispatch=task.early_dispatch_flag) for t, task in tasks.items()}
        consumer = "__consumer__"
        markers = module("deps_viewer")._task_markers(
            nodes | {consumer}, [(p, consumer) for _, p in pred_edges], {}, table
        )
        eligible = "⭐" in markers.get(consumer, "")
    if not eligible:
        return EarlyProof(NONE, None, None, 0, len(succ_rows))

    fins = [
        tasks[p].max_finish_us for _, p in pred_edges if p in tasks and tasks[p].max_finish_us is not None
    ]
    if not fins:
        return EarlyProof.unavailable()
    observed_ready = max(fins)
    tol_us = TOL_TICKS * 1_000_000.0 / float(freq_hz)

    proven = 0
    for row in succ_rows:
        if row.dispatch_us is not None and row.dispatch_us + tol_us < observed_ready:
            proven += 1
    total = len(succ_rows)
    if proven == total:
        return EarlyProof(FULL, observed_ready, tol_us, proven, total)
    if proven > 0:
        return EarlyProof(PARTIAL, observed_ready, tol_us, proven, total)
    return EarlyProof(NONE, observed_ready, tol_us, 0, total)


def classify(
    succ_rows: Sequence[RowSlice],
    pred_edges: Sequence[tuple[str, str]],
    tasks: Mapping[str, TaskTiming],
    freq_hz: int | None,
    level: int,
    eligible: bool | None = None,
) -> str:
    """Backward-compatible status-only form (T3 tests use this)."""
    return prove(succ_rows, pred_edges, tasks, freq_hz, level, eligible).status
