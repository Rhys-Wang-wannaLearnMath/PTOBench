# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""scope_stats metadata parser (DESIGN.md T9).

Reads ``scope_stats/scope_stats.jsonl`` — line 1 is a resource-capacity
metadata object, each later line is a per-scope ``begin``/``end`` record
— and returns one row per line. The first row (seq 0) holds the metadata
object; every other row keeps ``site``/``ring``/``phase`` as columns and
the remaining fields in a JSON payload.
"""

from __future__ import annotations

import json
from typing import Any

_META_KEYS = ("site", "ring", "phase")


def parse_scope_stats(text: str) -> list[dict[str, Any]]:
    lines = [line for line in text.splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    if not lines:
        return rows
    meta = json.loads(lines[0]) if lines else {}
    rows.append({"seq": 0, "site": None, "ring": None, "phase": None, "payload": meta if isinstance(meta, dict) else {}})
    seq = 1
    for line in lines[1:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        rows.append(
            {
                "seq": seq,
                "site": record.get("site"),
                "ring": record.get("ring"),
                "phase": record.get("phase"),
                "payload": {k: v for k, v in record.items() if k not in _META_KEYS},
            }
        )
        seq += 1
    return rows
