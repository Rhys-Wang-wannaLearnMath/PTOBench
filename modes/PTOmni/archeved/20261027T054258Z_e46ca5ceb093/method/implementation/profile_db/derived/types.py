# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared record types of the derived layer (DESIGN.md 10.2).

Every derivator is a pure function: tables/streams in, row lists out.
These frozen dataclasses are the only in-memory shapes they exchange;
the derived package never touches the ingest parsers or the raw JSON
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class RowSlice:
    """One physical execution row (task_row + per-row dispatch columns)."""

    task_id: str
    core_index: int
    engine: str
    start_us: float
    end_us: float
    dispatch_us: float | None
    receive_us: float | None
    finish_us: float | None


@dataclass(frozen=True)
class TaskTiming:
    """One logical task's stored timing aggregates (task table row)."""

    task_id: str
    engine: str | None
    early_dispatch_flag: bool
    num_rows: int
    busy_us: float | None
    wall_us: float | None
    min_dispatch_us: float | None
    min_receive_us: float | None
    min_start_us: float | None
    max_end_us: float | None
    max_finish_us: float | None


@dataclass(frozen=True)
class EdgeRec:
    """One dependency edge reduced to the fields derivation uses. Rows are
    ordered by edge_id, which follows the deps.json file order — the same
    order the upstream critical-path tool consumes, keeping hb-edge
    retention deterministic."""

    pred: str
    succ: str
    source: str


@dataclass(frozen=True)
class DerivedResult:
    """Everything ``derive_run`` computes for one run, ready for the writer."""

    bands: Sequence[dict[str, Any]]
    gaps: Sequence[dict[str, Any]]
    paths: Sequence[dict[str, Any]]  # cpm_path rows incl. the kind column
    cpm_us: float | None
    task_flags: Sequence[dict[str, Any]]  # {task_id, on_cpm_observed, on_cpm_static}


@dataclass(frozen=True)
class GapWindow:
    """Drain-suffix anchor handed from the band builder to the gap classifier:
    one engine's band geometry within the shared run axis."""

    t0_us: float
    width_us: float
    band_ends: tuple[float, ...]  # t1_us of each band, ascending
    drain_first: int | None  # first band index in the drain suffix (None = none)


def num_key(value: str) -> tuple[int, int, str]:
    """Deterministic sort key: numeric task ids order numerically, anything
    else falls back to lexicographic order."""
    try:
        return (0, int(value), "")
    except ValueError:
        return (1, 0, value)