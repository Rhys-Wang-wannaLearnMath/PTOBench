# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate capture identity and project single executions onto upstream formats."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any
from collections import Counter

from profile_db.errors import IngestError
from profile_db.runtime_adapter.backend import parser_keys, require, BackendUnavailable


def _load_converter():
    try:
        read_perf_data = require("swimlane_converter", "read_perf_data").read_perf_data
    except BackendUnavailable as exc:  # pragma: no cover - depends on the pypto env
        raise IngestError(
            "simpler_setup is unavailable; capture levels 2-4 require the "
            "PyPTO environment for the AICore<->AICPU clock-domain join"
        ) from exc
    return read_perf_data


def _materialize_chip_spelling(records_path: Path, records: dict[str, Any]) -> tuple[Path, bool]:
    """Return an upstream v1 projection; never modify the registered artifact.

    Callers must validate raw execution identities before projecting epochs.
    """
    normalized = copy.deepcopy(records)
    if "chip_swimlane_level" not in normalized:
        normalized["chip_swimlane_level"] = normalized.pop("l2_swimlane_level", None)
    # A single validated execution can be projected onto the upstream v1
    # wire format. Epochs remain in the original registered artifact.
    normalized["aicore_tasks"] = [row[:6] for row in records.get("aicore_tasks", [])]
    for key in ("scheduler_tasks", "scheduler_records"):
        if key in normalized:
            normalized[key].setdefault("schema_version", 1)
    if "scheduler_tasks" in normalized:
        normalized["scheduler_tasks"]["records"] = [row[:4] for row in normalized["scheduler_tasks"]["records"]]
    if "scheduler_records" in normalized:
        for stream in normalized["scheduler_records"]["streams"]:
            for record in stream["records"]:
                record.pop("run_epoch", None)
    if normalized == records:
        return records_path, False
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_pfdb_chip.json", prefix="pfdb_ingest_", delete=False
    )
    json.dump(normalized, tmp)
    tmp.close()
    return Path(tmp.name), True


# Scheduler records carry these fields and, on identity-stamped captures,
# run_epoch. Every record in a stream has the same field set.
_SCHEDULER_RECORD_FIELDS = frozenset(
    {"start_cycles", "end_cycles", "loop_iter", "kind", "tasks_processed", "task_id"}
)


def _merge_scheduler_streams(section: Any) -> list[list[dict[str, Any]]]:
    """``scheduler_records{streams}`` -> per-stream lanes
    with each stream's ``metrics`` merged onto its record by
    ``record_index``. Validation mirrors the upstream converter so a
    corrupt stream fails loudly instead of shifting phase rows."""
    if not isinstance(section, dict):
        raise IngestError("scheduler_records must be an object")
    if section.get("schema_version", 1) != 1:
        raise IngestError("unsupported scheduler_records schema_version; expected 1")
    streams = section.get("streams")
    if not isinstance(streams, list):
        raise IngestError("scheduler_records.streams must be an array")
    lanes: list[list[dict[str, Any]]] = []
    for stream_index, stream in enumerate(streams):
        if not isinstance(stream, dict):
            raise IngestError(f"scheduler_records.streams[{stream_index}] must be an object")
        records = stream.get("records")
        metrics = stream.get("metrics") or []
        if not isinstance(records, list) or not isinstance(metrics, list):
            raise IngestError(f"scheduler stream {stream_index} records/metrics must be arrays")
        record_fields = _SCHEDULER_RECORD_FIELDS
        if records and isinstance(records[0], dict) and "run_epoch" in records[0]:
            record_fields = record_fields | {"run_epoch"}
        merged: list[dict[str, Any]] = []
        for record_index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != record_fields:
                raise IngestError(
                    f"scheduler stream {stream_index} record {record_index} must contain exactly "
                    f"{sorted(record_fields)}"
                )
            if int(record["end_cycles"]) < int(record["start_cycles"]):
                raise IngestError(f"scheduler stream {stream_index} record {record_index} has a negative interval")
            merged.append(dict(record))
        for metric in metrics:
            if not isinstance(metric, dict) or "record_index" not in metric:
                raise IngestError(f"scheduler stream {stream_index} has malformed metrics")
            record_index = int(metric["record_index"])
            if record_index < 0 or record_index >= len(merged):
                raise IngestError(
                    f"scheduler stream {stream_index} metric record_index {record_index} is out of range"
                )
            metric_values = {key: value for key, value in metric.items() if key != "record_index"}
            overwritten = set(metric_values) & (record_fields | {"run_epoch"})
            if overwritten:
                raise IngestError(
                    f"scheduler stream {stream_index} metric overwrites fixed record fields: {sorted(overwritten)}"
                )
            merged[record_index].update(metric_values)
        lanes.append(merged)
    return lanes


def _scheduler_phase_lanes(records: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Raw scheduler phase lanes for the level-1 local path, in file order.

    Levels 2-4 use the converter's scheduler-ID-indexed lanes instead.
    """
    section = records.get("scheduler_records")
    if section is not None:
        return _merge_scheduler_streams(section)
    return [list(lane) for lane in records.get("aicpu_scheduler_phases") or []]


def validate_capture(records: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Validate raw identities before a backend can discard any records."""
    raw_rows = records.get("aicore_tasks") or []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, list) or len(row) not in (5, 6, 7):
            raise IngestError(f"aicore_tasks[{index}] must contain five, six or seven columns")
    epochs = {int(row[6]) if len(row) == 7 else None for row in raw_rows}
    section = records.get("scheduler_tasks")
    if section is not None:
        if "aicpu_tasks" in records:
            raise IngestError("both scheduler_tasks and legacy aicpu_tasks are present")
        if not isinstance(section, dict) or section.get("schema_version", 1) != 1:
            raise IngestError("unsupported scheduler_tasks schema_version; expected 1")
        if section.get("producer") not in ("aicpu", "aicore"):
            raise IngestError("scheduler_tasks.producer must be aicpu or aicore")
        if not isinstance(section.get("records"), list):
            raise IngestError("scheduler_tasks.records must be an array")
    timing_rows = section["records"] if section is not None else records.get("aicpu_tasks") or []
    timing_epochs = set()
    for row in timing_rows:
        if not isinstance(row, list) or len(row) not in (4, 5):
            raise IngestError("scheduler task rows must contain four or five columns")
        timing_epochs.add(int(row[4]) if len(row) == 5 else None)
    if len(timing_epochs) > 1:
        raise IngestError("scheduler tasks contain mixed run_epoch values")
    epochs.update(e for e in timing_epochs if e is not None)
    if "scheduler_records" in records and records.get("aicpu_scheduler_phases"):
        raise IngestError("both scheduler_records and legacy scheduler phases are present")
    lanes = _scheduler_phase_lanes(records)
    for lane in lanes + list(records.get("aicpu_orchestrator_phases") or []) + list(records.get("host_orchestrator_phases") or []):
        phase_epochs = set()
        for phase in lane:
            if not isinstance(phase, dict):
                raise IngestError("phase records must be objects")
            if phase.get("run_epoch") is not None:
                epochs.add(int(phase["run_epoch"]))
            phase_epochs.add(phase.get("run_epoch"))
        if len(phase_epochs) > 1:
            raise IngestError("phase stream contains mixed run_epoch values")
    for record in records.get("aicpu_lifecycle_records") or []:
        if not isinstance(record, dict):
            raise IngestError("aicpu_lifecycle_records must contain objects")
        if record.get("run_epoch") is not None:
            epochs.add(int(record["run_epoch"]))
    if len(epochs) > 1:
        raise IngestError(f"capture contains multiple or mixed run_epoch values {epochs}; ingest one execution per capture")
    return lanes


def read_capture(records_path: Path, records=None, *, calls=None):
    """Join with the selected backend, rejecting unsupported or lost input.

    Format projection is local; cross-clock calculations remain upstream.
    This path is shared by ingest, scheduler analysis and trace export.
    """
    records_path = Path(records_path)
    records = json.loads(records_path.read_text()) if records is None else records
    lanes = validate_capture(records)
    converter = _load_converter()
    keys = parser_keys(converter)
    for key in ("scheduler_tasks", "scheduler_records", "aicpu_lifecycle_records"):
        if records.get(key) and key not in keys:
            raise IngestError(f"selected Simpler converter does not support {key}; use the matching PyPTO runtime")
    path, temporary = _materialize_chip_spelling(records_path, records)
    try:
        if calls is not None:
            if temporary:
                calls.compatibility.append("capture-single-epoch-v1")
            joined = calls.call("swimlane_converter", "read_perf_data", str(path))
        else:
            joined = converter(str(path))
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise IngestError(f"upstream converter rejected the capture: {exc}") from exc
    finally:
        if temporary:
            path.unlink(missing_ok=True)
    if len(joined.get("tasks", [])) != len(records.get("aicore_tasks") or []):
        raise IngestError("upstream converter lost or duplicated AICore task rows")
    if Counter((int(t["core_id"]), int(t["task_id"])) for t in joined.get("tasks", [])) != Counter(
        (int(row[0]), int(row[1])) for row in records.get("aicore_tasks") or []
    ):
        raise IngestError("upstream converter changed AICore task identity")
    for key, expected in (("aicpu_scheduler_phases", lanes),
                          ("aicpu_orchestrator_phases", records.get("host_orchestrator_phases") or records.get("aicpu_orchestrator_phases") or [])):
        def counts(lanes):
            return {i: len(lane) for i, lane in enumerate(lanes) if lane}
        if counts(joined.get(key, [])) != counts(expected):
            raise IngestError(f"upstream converter lost or duplicated {key} records")
    return joined

