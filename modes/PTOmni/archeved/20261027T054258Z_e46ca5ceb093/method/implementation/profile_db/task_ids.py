# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Task-id normalization shared by capture parsers and writers.

The runtime emits one unsigned 64-bit task identity in several spellings:
the graph and swimlane normally use decimal while PMU commonly uses ``0x``
hexadecimal.  Queries join on the canonical decimal spelling, but the original
token remains available as provenance.  Host creator identifiers are not task
numbers and deliberately remain opaque strings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskId:
    """A lossless task identifier with an optional unsigned-64 key."""

    canonical: str
    raw: str
    u64: str | None


def normalize_task_id(value: object) -> TaskId:
    """Return the canonical decimal spelling when *value* is a U64 token.

    Decimal and ``0x`` hexadecimal spellings are accepted.  Whitespace,
    negative values, overflows, and nonnumeric creator identifiers retain their
    exact string spelling and have no numeric join key.
    """
    raw = str(value).strip()
    if not raw:
        return TaskId(canonical=raw, raw=raw, u64=None)
    base = 16 if raw.lower().startswith("0x") else 10
    try:
        parsed = int(raw, base)
    except ValueError:
        return TaskId(canonical=raw, raw=raw, u64=None)
    if parsed < 0 or parsed > (2**64 - 1):
        return TaskId(canonical=raw, raw=raw, u64=None)
    canonical = str(parsed)
    return TaskId(canonical=canonical, raw=raw, u64=canonical)
