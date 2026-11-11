# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Recorded scheduler activity, rebuilt from retained originals on every read.

Legacy phase arrays do not certify drop-free coverage or identify the async
engine/task being polled. Interval unions describe recorded activity only.
"""
from collections import defaultdict
import json
import math

from profile_db.errors import QueryError
from profile_db.runtime_adapter.capture import read_capture, validate_capture
from profile_db.runtime_adapter.storage import resolve

VERSION = 'scheduler_observations_v2'
PHASES = frozenset(('complete', 'async_poll', 'dispatch', 'release', 'dummy',
                   'early_dispatch', 'drain', 'graph_prepare', 'resolve', 'resolve_standalone'))
OPTIONAL_COUNTS = ('loop_iter', 'tasks_processed', 'pop_hit', 'pop_miss')


def _counts(record):
    counts = {k: record[k] for k in OPTIONAL_COUNTS if k in record}
    # Archived captures stored these two fields as flags, not integer counts.
    for key in ('pop_hit', 'pop_miss'):
        if type(counts.get(key)) is bool:
            counts[key] = int(counts[key])
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise QueryError('scheduler_phase_capture_invalid')
    return counts


def _union(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        total += stop-start if end is None or start > end else max(0.0, stop-end)
        end = stop if end is None else max(end, stop)
    return total


def analyze(conn, run_id):
    """Return capture-local phase rows and polling metrics; never SQL caches."""
    kinds = [r[0] for r in conn.execute(
        "SELECT kind FROM artifact WHERE run_id=? AND kind IN "
        "('chip_swimlane_records','l2_swimlane_records','l2_perf_records')", [run_id]).fetchall()]
    if len(kinds) != 1:
        raise QueryError('scheduler_phase_capture_absent')
    path = resolve(conn, run_id, kinds[0])
    raw = json.loads(path.read_text())
    level = raw.get('chip_swimlane_level', raw.get('l2_swimlane_level'))
    if type(level) is not int or level not in (3, 4):
        raise QueryError('scheduler_phase_capture_absent')
    if not any(raw.get(k) is not None for k in ('aicpu_scheduler_phases', 'scheduler_records')):
        raise QueryError('scheduler_phase_capture_absent')
    legacy = raw.get('aicpu_scheduler_phases')
    if legacy is not None and (not isinstance(legacy, list) or
                               any(not isinstance(lane, list) for lane in legacy)):
        raise QueryError('scheduler_phase_capture_invalid')
    original = validate_capture(raw)
    for lane in original:
        for record in lane:
            start, end = record.get('start_cycles'), record.get('end_cycles')
            if (type(start) is not int or type(end) is not int or start < 0 or end < start
                    or not isinstance(record.get('kind'), str)):
                raise QueryError('scheduler_phase_capture_invalid')
            _counts(record)
    joined = read_capture(path, raw)
    converted = joined.get('aicpu_scheduler_phases') or []
    if (len(converted) != len(original) or
            any(len(a) != len(b) for a, b in zip(converted, original))):
        raise QueryError('scheduler_phase_capture_invalid')
    rows, polling = [], defaultdict(list)
    for lane, records in enumerate(converted):
        for ordinal, record in enumerate(records):
            start, end = record.get('start_time_us'), record.get('end_time_us')
            phase = record.get('phase')
            if (not isinstance(phase, str) or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end))
                    or end < start or not math.isfinite(end-start)):
                raise QueryError('scheduler_phase_capture_invalid')
            if phase != original[lane][ordinal]['kind']:
                raise QueryError('scheduler_phase_capture_invalid')
            counts = _counts(record)
            if counts != _counts(original[lane][ordinal]):
                raise QueryError('scheduler_phase_capture_invalid')
            rows.append(dict(entity=f'{lane}:{ordinal}', scheduler_lane=lane, ordinal=ordinal,
                             phase=phase if phase in PHASES else 'unrecognized',
                             start_us=start, end_us=end, duration_us=end-start, **counts))
            if phase == 'async_poll':
                polling[lane].append((start, end))
    values = dict(recorded_async_poll_records=sum(map(len, polling.values())),
                  recorded_async_poll_lanes=len(polling),
                  recorded_async_poll_lane_us=sum(_union(spans) for spans in polling.values()),
                  recorded_async_poll_union_us=_union([span for spans in polling.values() for span in spans]))
    if any(not math.isfinite(v) for v in values.values()):
        raise QueryError('scheduler_phase_capture_invalid')
    return dict(rows=rows, metrics=values, parser_version=VERSION,
                interpretation='recorded_scheduler_activity',
                capture_completeness='uncertified', task_engine_attribution=False,
                dma_wait_time=False, performance_score=False)
