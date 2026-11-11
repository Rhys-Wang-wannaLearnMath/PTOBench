# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parsers for the text-modality evidence files (T2).

All parsers are pure functions: text in, row dicts out. Compiler-origin
text (perf hints) is preserved verbatim; the environment metadata we
invent ourselves (runtime_cfg) is the only thing redacted, and user
notes/tags are stored as supplied.

Formats (calibrated on build_output artifacts):

- ``report/perf_hints.log``: one hint per line,
  ``[perf_hint PH-XXX] <message> at <path>:<line>:<col>``;
- ``report/memory_after_AllocateMemoryAddr.txt``: ``--- <kernel> ---``
  headings followed by pipe rows ``space | used | limit | pct | memrefs``
  (Vec/Mat/Left/Right/Acc per DESIGN.md 5.2);
- ``dfx_outputs/pmu.csv``: header row with a task-id column plus
  dynamic per-pipe counter columns; melted into long form;
- benchmark: the ``effective_us (N rounds) min=.. median=.. mean=.. max=..``
  line printed by the ``PYPTO_BENCH`` loop, or an equivalent
  ``key=value`` string.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
from typing import Any

from profile_db.errors import IngestError
from profile_db.task_ids import normalize_task_id

_HINT_SOURCE = re.compile(r"^(?P<text>.*) at (?P<source>\S+:\d+:\d+)\s*$")
_HEADING = re.compile(r"^---\s+(.+?)\s+---$")
_SIZE_CELL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)?\s*$", re.IGNORECASE)
_BENCH_ROUNDS = re.compile(r"\((\d+)\s+rounds?\)")
_BENCH_KEYS = {"min", "median", "mean", "max"}
_RAW_HEADER = re.compile(
    r"^\[RUN\]\s+raw samples:\s+ranks=(?P<ranks>\d+)\s+rounds=(?P<rounds>\d+)\s+"
    r"warmup=(?P<warmup>\d+)(?P<fallback>\s+fallback_flattened=1)?\s*$"
)
_RAW_HEADLINE = re.compile(r"^\[RUN\]\s+headline raw n=(?P<count>\d+)\s+eff_us=(?P<samples>\[.*\])\s*$")
_ABS_PATH = re.compile(r"/(?:(?:\w+/)?home|root|data\d+)/[^/\s]+")
# The total-cycles column of pmu.csv, matched by shape because the exact
# header varies by architecture (DESIGN.md 5.2: PMU column names are dynamic).
_TOTAL_CYCLES = re.compile(r"^(?=.*total)(?=.*cycle).*$", re.IGNORECASE)
_PMU_COORDINATES = {
    "thread_id",
    "thread",
    "core_id",
    "core",
    "func_id",
    "function_id",
    "core_type",
    "event_type",
}


def parse_perf_hints(text: str) -> list[dict[str, Any]]:
    """One row per non-empty line; the trailing `` at <path>:<line>:<col>``
    suffix is split into ``source_path``, the message stays verbatim in
    ``text`` (origin is always compiler)."""
    rows = []
    for seq, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip():
            continue
        match = _HINT_SOURCE.match(line)
        rows.append(
            {
                "seq": seq,
                "text": line,
                "source_path": match.group("source") if match else None,
                "origin": "compiler",
            }
        )
    return rows


def parse_bytes(value: str) -> float | None:
    """``32768 B`` / ``32KB`` -> bytes; None for unrecognized cells."""
    match = _SIZE_CELL.match(value.strip())
    if match is None:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
    return number * scale


def parse_memory_report(text: str) -> list[dict[str, Any]]:
    """Pipe-row buffer occupancy under each ``--- <kernel> ---`` heading.

    Columns: space | used | limit | percent | memrefs. Stored: kernel,
    space, usage (used bytes), limit_value (limit bytes); the percent and
    memref columns carry no extra semantics beyond the bytes and are
    dropped by design (raw text stays in the artifact).
    """
    rows: list[dict[str, Any]] = []
    kernel = "unknown"
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        heading = _HEADING.match(line)
        if heading:
            kernel = heading.group(1)
            continue
        if not line:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        cells = [cell for cell in cells if cell]
        if len(cells) < 3 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", cells[0]):
            continue
        used = parse_bytes(cells[1])
        limit = parse_bytes(cells[2])
        if used is None or limit is None:
            raise IngestError(
                f"memory report line {lineno}: cannot parse bytes cells: {line!r}"
            )
        rows.append(
            {
                "kernel": kernel,
                "space": cells[0],
                "usage": used,
                "limit_value": limit,
            }
        )
    return rows


def parse_pmu_csv(text: str) -> list[dict[str, Any]]:
    """Melt a dynamic-column pmu.csv into long form: one row per
    (task, counter) measurement, the task token preserved verbatim.

    One column per task carries the task's total cycle count; it is
    recognized by name shape (``*total*cycle*``, e.g.
    ``pmu_total_cycles``) rather than an exact spelling, because the
    header varies by architecture. Its value is stamped onto every row of
    that task as ``total_cycles`` so the query layer can report a
    per-pipeline occupancy ratio; the column itself is still melted into
    its own counter row, so nothing from the artifact is dropped.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise IngestError("pmu.csv has no header row")
    fieldnames = [name.strip() for name in reader.fieldnames]
    if not all(fieldnames) or len({name.lower() for name in fieldnames}) != len(fieldnames):
        raise IngestError('pmu.csv has duplicate or empty column names')
    reader.fieldnames = fieldnames
    id_columns = [name for name in fieldnames if name.lower() in ("task_id", "task", "id")]
    if len(id_columns) != 1:
        raise IngestError(
            f"pmu.csv must have exactly one task id column; header: {fieldnames}"
        )
    id_column = id_columns[0]
    columns_by_lower = {name.strip().lower(): name for name in fieldnames}
    counter_columns = [
        name for name in fieldnames
        if name != id_column and name.strip().lower() not in _PMU_COORDINATES
    ]
    if not counter_columns:
        raise IngestError("pmu.csv carries no counter columns besides the task id")
    total_columns = [name for name in counter_columns if _TOTAL_CYCLES.match(name)]
    total_column = total_columns[0] if len(total_columns) == 1 else None

    rows: list[dict[str, Any]] = []
    for lineno, record in enumerate(reader, start=2):
        if None in record or any(value is None for value in record.values()):
            raise IngestError(f'pmu.csv line {lineno}: row width does not match header')
        task_id_raw = (record.get(id_column) or "").strip()
        if not task_id_raw:
            raise IngestError(f"pmu.csv line {lineno}: empty task id")
        task_id = normalize_task_id(task_id_raw)

        def _integer_coordinate(*names: str) -> int | None:
            column = next((columns_by_lower.get(name) for name in names if name in columns_by_lower), None)
            if column is None:
                return None
            raw = (record.get(column) or "").strip()
            if not raw:
                return None
            try:
                return int(raw, 0)
            except ValueError as exc:
                raise IngestError(
                    f"pmu.csv line {lineno} column {column!r}: non-integer value {raw!r}"
                ) from exc

        def _value(column: str) -> float | None:
            raw = (record.get(column) or "").strip()
            if not raw:
                return None  # sparse cells are absence, not zero
            try:
                value = float(raw)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("counter must be finite and nonnegative")
                return value
            except ValueError as exc:
                raise IngestError(
                    f"pmu.csv line {lineno} column {column!r}: "
                    f"non-numeric value {raw!r}"
                ) from exc

        total = _value(total_column) if total_column is not None else None
        for counter in counter_columns:
            value = _value(counter)
            if value is None:
                continue
            rows.append(
                {
                    "task_id": task_id.canonical,
                    "task_id_raw": task_id.raw,
                    "task_id_u64": task_id.u64,
                    "sample_seq": lineno - 2,
                    "thread_id": _integer_coordinate("thread_id", "thread"),
                    "core_id": _integer_coordinate("core_id", "core"),
                    "func_id": _integer_coordinate("func_id", "function_id"),
                    "core_type": (record.get(columns_by_lower.get("core_type", "")) or "").strip() or None,
                    "event_type": (record.get(columns_by_lower.get("event_type", "")) or "").strip() or None,
                    "counter": counter,
                    "value": value,
                    "total_cycles": total,
                }
            )
    return rows


def parse_bench_line(text: str) -> dict[str, Any]:
    """Extract bench summary from the PYPTO_BENCH output line (or an
    equivalent ``min=.. median=.. mean=.. max=..`` string)."""
    fields: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=([^ \t]+)", text):
        fields[match.group(1)] = match.group(2)
    missing = _BENCH_KEYS - set(fields)
    if missing:
        raise IngestError(f"bench line is missing {sorted(missing)}: {text!r}")
    try:
        parsed = {key: float(fields[key]) for key in _BENCH_KEYS}
    except ValueError as exc:
        raise IngestError(f"bench line carries non-numeric values: {text!r}") from exc
    if "rounds" in fields:
        try:
            parsed["rounds"] = int(fields["rounds"])
        except ValueError as exc:
            raise IngestError(f"bench rounds is not an integer: {fields['rounds']!r}") from exc
    else:
        rounds_match = _BENCH_ROUNDS.search(text)
        parsed["rounds"] = int(rounds_match.group(1)) if rounds_match else None
    return parsed


def parse_bench_log(text: str) -> dict[str, Any]:
    """Parse one benchmark invocation including its authoritative raw samples.

    The harness emits one ``headline raw`` sequence per invocation.  Keeping
    this as one stratum preserves which invocation each sample came from
    instead of merging separately launched benchmark processes.
    """
    summary_line = next(
        (line for line in text.splitlines() if "effective_us" in line and all(f"{key}=" in line for key in _BENCH_KEYS)),
        None,
    )
    if summary_line is None:
        raise IngestError("bench log is missing an effective_us summary line")
    summary = parse_bench_line(summary_line)
    header = next((_RAW_HEADER.match(line.strip()) for line in text.splitlines() if _RAW_HEADER.match(line.strip())), None)
    headline = next((_RAW_HEADLINE.match(line.strip()) for line in text.splitlines() if _RAW_HEADLINE.match(line.strip())), None)
    if header is None or headline is None:
        raise IngestError(
            "bench log is missing authoritative raw samples; rerun with PYPTO_BENCH_RAW=1"
        )
    if header.group("fallback"):
        raise IngestError("bench log uses fallback_flattened samples and is not stratifiable")
    try:
        samples = json.loads(headline.group("samples"))
    except json.JSONDecodeError as exc:
        raise IngestError("bench raw headline has invalid JSON samples") from exc
    if not isinstance(samples, list) or not samples or not all(
        isinstance(sample, (int, float)) and not isinstance(sample, bool) for sample in samples
    ):
        raise IngestError("bench raw headline must contain a non-empty numeric sample list")
    if len(samples) != int(headline.group("count")):
        raise IngestError("bench raw headline count does not match its sample list")
    if int(header.group("rounds")) != len(samples):
        raise IngestError("bench raw header rounds does not match headline samples")
    if summary.get("rounds") is not None and int(summary["rounds"]) != len(samples):
        raise IngestError("bench summary rounds does not match authoritative raw samples")
    values = [float(sample) for sample in samples]
    return {
        "samples": values,
        "rounds": len(values),
        "warmup": int(header.group("warmup")),
        "rank_count": int(header.group("ranks")),
        "aggregation_mode": "headline_effective",
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def redact_paths(value: Any) -> Any:
    """Recursively replace absolute path tokens (``/home/user/...``,
    ``/data1/user/...``, ``/root/...``) in environment metadata with
    ``/<redacted>``. Compiler-origin data never passes through here."""
    if isinstance(value, dict):
        return {str(key): redact_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_paths(item) for item in value]
    if isinstance(value, str):
        return _ABS_PATH.sub("/<redacted>", value)
    return value


def redact_json(value: Any) -> str:
    """JSON text with paths redacted (input to CAST(? AS JSON))."""
    return json.dumps(redact_paths(value or {}))
