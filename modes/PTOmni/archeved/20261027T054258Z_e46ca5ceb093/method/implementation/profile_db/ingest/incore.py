# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""In-core manifest metadata; existing traces are registered by generic attachment."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

# CSV columns kept in the metrics JSON (dedicated columns excluded).
_METRIC_COLS = (
    "artifact_count",
    "core_trace_count",
    "instr_csv_count",
    "duration_sec",
    "message",
    "symbol",
    "demangled",
    "app",
    "kernel_lib",
)


def parse_manifest(text: str) -> list[dict[str, Any]]:
    """Parse ``manifest_export.csv`` text into incore-entry row dicts."""
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    for record in reader:
        metrics: dict[str, Any] = {}
        for key in _METRIC_COLS:
            value = record.get(key)
            if value not in ("", None):
                metrics[key] = value
        rows.append(
            {
                "kernel": record.get("func"),
                "status": record.get("status"),
                "export_dir": record.get("export_dir"),
                "metrics": metrics,
            }
        )
    return rows


def merge_instr_metrics(rows: list[dict[str, Any]], instr_text: str | None) -> None:
    """Attach an ``instr_metrics.json`` summary to every exported row's
    metrics (the fixture keeps one summary beside the manifest)."""
    if not instr_text or not instr_text.strip():
        return
    try:
        summary = json.loads(instr_text)
    except (TypeError, ValueError):
        return
    for row in rows:
        if row.get("status") == "exported":
            row["metrics"]["instr_metrics"] = summary
