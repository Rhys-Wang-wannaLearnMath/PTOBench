# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Time-band density index derivation (DESIGN.md 5.3, T3).

The run axis ``[min(start), max(end)]`` over every physical row is cut
into fixed-width bands: storage granularity 5 µs, adaptively widened
(``max(5µs, span/10000)``) so an arbitrarily long run never explodes past
the 10k-band budget. For every engine, each band records how many of the
engine's cores are busy (any row overlapping the band interval) and which
tasks occupy it.

Determinations per band (5.3 decision table):

- ``sparse``: ``busy_cores <= 25% * cores`` of that engine, but never in
  the drain-tail suffix;
- ``drain_tail``: the suffix of bands after the last band whose busy core
  count reached 50% of that engine's cores; when no band ever reaches
  50% the suffix is empty.

Engine core counts come from the capture metadata (``run.core_types``);
only when the metadata lacks an engine present in the rows does the
builder fall back to the distinct cores observed for that engine. All
numbers derive from stored rows — nothing is hardcoded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from profile_db.derived.types import GapWindow, RowSlice, num_key

BAND_DEFAULT_US = 5.0
MAX_BANDS = 10_000


@dataclass(frozen=True)
class BandRow:
    """One (run, band, engine) density cell, ready for the writer."""

    band_idx: int
    t0_us: float
    t1_us: float
    engine: str
    total_cores: int
    busy_cores: int
    task_ids: tuple[str, ...]
    sparse: bool
    drain_tail: bool


def band_width_us(span_us: float) -> float:
    """Storage band width for a run: 5 µs, widened when the span would
    otherwise exceed the 10k-band budget."""
    return max(BAND_DEFAULT_US, span_us / MAX_BANDS)


def build_bands(
    rows: Sequence[RowSlice], core_types: Sequence[str]
) -> tuple[BandRow, ...]:
    """Density cells over the shared run axis; empty input yields no bands."""
    if not rows:
        return ()
    t0_us = min(row.start_us for row in rows)
    t1_us = max(row.end_us for row in rows)
    span = t1_us - t0_us
    if span <= 0:
        return ()
    width = band_width_us(span)
    n_bands = int(span / width) + 1
    if n_bands > MAX_BANDS + 1:
        # Defensive cap: width already keeps this from happening; a runaway
        # float width would only truncate the trailing bands here.
        n_bands = MAX_BANDS + 1

    engines = sorted({row.engine for row in rows})
    metadata_types = list(core_types)
    total = {
        engine: (
            metadata_types.count(engine)
            if engine in metadata_types
            else len({row.core_index for row in rows if row.engine == engine})
        )
        for engine in engines
    }

    busy: dict[tuple[str, int], set[int]] = {}
    occupied: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        first = max(0, math.floor((row.start_us - t0_us) / width))
        last = min(n_bands - 1, math.floor((row.end_us - t0_us) / width))
        for band in range(first, last + 1):
            band_t0 = t0_us + band * width
            band_t1 = t0_us + (band + 1) * width
            if row.start_us < band_t1 and row.end_us > band_t0:
                key = (row.engine, band)
                busy.setdefault(key, set()).add(row.core_index)
                ids = occupied.setdefault(key, [])
                if row.task_id not in ids:
                    ids.append(row.task_id)

    out: list[BandRow] = []
    for engine in engines:
        cores = total[engine]
        full_at = next(
            (
                band
                for band in range(n_bands - 1, -1, -1)
                if len(busy.get((engine, band), ())) >= cores * 0.5
            ),
            None,
        )
        drain_first = full_at + 1 if full_at is not None else None
        for band in range(n_bands):
            band_cores = len(busy.get((engine, band), ()))
            drain = drain_first is not None and band >= drain_first
            out.append(
                BandRow(
                    band_idx=band,
                    t0_us=t0_us + band * width,
                    t1_us=t0_us + (band + 1) * width,
                    engine=engine,
                    total_cores=cores,
                    busy_cores=band_cores,
                    task_ids=tuple(sorted(occupied.get((engine, band), ()), key=num_key)),
                    sparse=band_cores <= cores * 0.25 and not drain,
                    drain_tail=drain,
                )
            )
    return tuple(out)


def gap_windows(bands: Sequence[BandRow]) -> dict[str, GapWindow]:
    """Per-engine band geometry for the idle-gap classifier (drain suffix
    lookup, axis origin and width)."""
    windows: dict[str, GapWindow] = {}
    grouped: dict[str, list[BandRow]] = {}
    for band in bands:
        grouped.setdefault(band.engine, []).append(band)
    for engine, engine_bands in grouped.items():
        first_drain = next(
            (b.band_idx for b in engine_bands if b.drain_tail), None
        )
        windows[engine] = GapWindow(
            t0_us=engine_bands[0].t0_us,
            width_us=engine_bands[0].t1_us - engine_bands[0].t0_us,
            band_ends=tuple(b.t1_us for b in engine_bands),
            drain_first=first_drain,
        )
    return windows