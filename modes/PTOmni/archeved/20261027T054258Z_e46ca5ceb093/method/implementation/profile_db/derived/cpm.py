# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Critical paths use the dedicated runtime algorithm."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from profile_db.derived.types import EdgeRec, RowSlice
from profile_db.runtime_adapter import critical_path
from profile_db.runtime_adapter.storage import read_snapshot


@dataclass(frozen=True)
class StaticStep:
    """One task of the static (dependency-limited) longest path."""

    seq: int
    task_id: str
    busy_us: float  # end - start over the task's physical rows


@dataclass(frozen=True)
class ObservedStep:
    """One task of the observed blame walk with its frontier-sweep tiling."""

    seq: int
    task_id: str
    kind: str  # data-wait / core-wait / front-gap
    start_us: float
    end_us: float
    compute_us: float  # non-overlapped compute contributed on the path
    stall_us: float  # gap before this task on the path


@dataclass(frozen=True)
class PathFacts:
    """Both paths plus the span anchors and the static duration."""

    static: tuple[StaticStep, ...]
    observed: tuple[ObservedStep, ...]
    cpm_us: float | None  # static path duration in µs (None: no timed tasks)
    t0_us: float | None  # first row start over all engines
    t1_us: float | None  # last row end over all engines


def from_payload(payload, rows):
    if not payload or payload.get("cpm_us") is None or not rows:
        return PathFacts((), (), None, None, None)
    first = min(r.start_us for r in rows)
    scale = 1e6 / payload["freq_hz"]
    origin = payload["t0_ticks"]
    return PathFacts(
        tuple(StaticStep(seq=i, **s) for i, s in enumerate(payload["static"])),
        tuple(
            ObservedStep(
                i,
                s["task_id"],
                s["kind"],
                first + (s["start_ticks"] - origin) * scale,
                first + (s["end_ticks"] - origin) * scale,
                s["compute_us"],
                s["stall_us"],
            )
            for i, s in enumerate(payload["observed"])
        ),
        payload["cpm_us"],
        first,
        max(r.end_us for r in rows),
    )


def stored_paths(conn, run_id, rows):
    doc = read_snapshot(conn, run_id, "critical_path")
    return from_payload(doc.get("payload") if doc["status"] == "ok" else None, rows)


def compute_paths(rows: Sequence[RowSlice], edges: Sequence[EdgeRec], freq_hz: int) -> PathFacts:
    """Adapt table-only fixtures to the runtime; real captures retain raw integer ticks."""
    if not rows:
        return PathFacts((), (), None, None, None)
    scale = freq_hz / 1e6
    raw = dict(
        metadata=dict(clock_freq_hz=freq_hz),
        aicore_tasks=[
            [r.core_index, r.task_id, i, round(r.start_us * scale), round(r.end_us * scale), 0]
            for i, r in enumerate(rows)
        ],
    )
    deps = dict(
        tasks=[dict(task_id=t, kernel_ids=[]) for t in dict.fromkeys(r.task_id for r in rows)],
        edges=[dict(pred=e.pred, succ=e.succ) for e in edges],
    )
    with TemporaryDirectory(prefix="pfdb-table-cpm-") as work:
        p = Path(work)
        for name, doc in (("records.json", raw), ("deps.json", deps), ("names.json", {})):
            (p / name).write_text(json.dumps(doc))
        payload = critical_path(p / "records.json", p / "deps.json", p / "names.json")
    return from_payload(payload, rows)
