# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fact DSL v2: machine-oriented, line-based evidence records.

Design contract (DESIGN.md 5.4):

- One record per line: ``REC k=v ... evidence=<state>``, keys sorted,
  values JSON-encoded, deterministic byte output.
- Every fact carries exactly one evidence state
  (``measured / proven / unproven / unavailable``). The enum rejects
  anything else at construction time.
- Serialization is budget-bound: when ``max_bytes`` is exhausted the
  stream ends with an explicit ``TRUNCATED first_dropped_index=...
  remaining=... limit=... hint="retry --budget N"`` line. The cut is a
  prefix, so what survives is always a contiguous head of the sequence;
  omitted facts are never silently dropped.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from profile_db.errors import FactError

_REC_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_RESERVED_FIELDS = frozenset({"evidence"})

_MISSING = object()


class Evidence(str, enum.Enum):
    """Evidence state attached to every fact (DESIGN.md 3.2)."""

    MEASURED = "measured"
    PROVEN = "proven"
    UNPROVEN = "unproven"
    UNAVAILABLE = "unavailable"


def _check_jsonable(value: Any) -> str:
    try:
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise FactError(f"fact value for a field must be JSON-encodable: {exc}") from exc
    return value


@dataclass(frozen=True)
class Fact:
    """One bounded evidence record. ``evidence`` is mandatory."""

    rec: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    evidence: Evidence = _MISSING  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.evidence is _MISSING:
            raise TypeError("Fact.evidence is required")
        if not isinstance(self.evidence, Evidence):
            raise TypeError(f"evidence must be an Evidence, got {type(self.evidence)!r}")
        if not _REC_NAME.match(self.rec):
            raise FactError(f"invalid record type {self.rec!r} (must match {_REC_NAME.pattern})")
        for key in self.fields:
            if not _FIELD_NAME.match(key):
                raise FactError(f"invalid field name {key!r}")
            if key in _RESERVED_FIELDS:
                raise FactError(f"field name {key!r} is reserved")
        # Freeze values and validate JSON-encodability eagerly. The mapping
        # proxy makes the frozen dataclass genuinely immutable.
        validated = MappingProxyType({key: _check_jsonable(value) for key, value in self.fields.items()})
        object.__setattr__(self, "fields", validated)


def format_fact(fact: Fact) -> str:
    """Serialize one fact to its canonical single line (no trailing newline)."""
    parts: list[str] = [fact.rec]
    for key in sorted(fact.fields):
        parts.append(f"{key}={json.dumps(fact.fields[key], ensure_ascii=False, separators=(',', ':'))}")
    parts.append(f"evidence={fact.evidence.value}")
    return " ".join(parts)


def truncated_line(dropped: int, first_dropped_index: int, max_bytes: int) -> str:
    """The canonical truncation marker. ``first_dropped_index`` is the
    0-based position of the first omitted fact; because truncation is a
    prefix cut, everything from there on is what is missing.

    ``hint`` names a budget that is at least 8× the exhausted limit and
    at least 32768 bytes — the size that typically admits ``pmu`` and
    ``critical_path`` streams.
    """
    hint = max(max_bytes * 8, 32768)
    return (
        f"TRUNCATED first_dropped_index={first_dropped_index} "
        f"remaining={dropped} limit={max_bytes} hint=\"retry --budget {hint}\""
    )


def truncate_facts(facts: Iterable[Fact], max_bytes: int = 4096) -> tuple[list[Fact], int]:
    """Prefix-truncate ``facts`` to the UTF-8 byte budget of the canonical
    facts rendering; returns ``(kept, dropped)``.

    The cut is a **prefix**: the first fact that does not fit ends the
    stream. Skipping an oversized fact and letting a later, shorter one
    take its place would hand an ordered sequence (a critical path, a row
    list) back with invisible holes, which reads as consecutive. Room for
    the ``TRUNCATED`` line is reserved before any fact is admitted, so the
    rendered output stays inside the budget — the sole exception is a
    budget smaller than the marker itself, where the marker is still
    emitted because silent omission is never an option.

    All output formats share this function, so ``json`` and ``markdown``
    are bounded by the same fact count as ``facts``.
    """
    if max_bytes < 1:
        raise FactError("max_bytes must be at least 1")
    items = list(facts)
    lines = [format_fact(fact) for fact in items]
    total = sum(len(line.encode("utf-8")) for line in lines) + max(len(lines) - 1, 0)
    if total <= max_bytes:
        return items, 0

    used = 0
    kept = 0
    for line in lines:
        cost = len(line.encode("utf-8")) + (1 if kept else 0)
        # The marker grows with the counts it reports, so its size is
        # re-measured for the hypothetical "admit this one" state.
        reserve = len(
            truncated_line(len(items) - kept - 1, kept + 1, max_bytes).encode("utf-8")
        ) + 1
        if used + cost + reserve > max_bytes:
            break
        used += cost
        kept += 1
    return items[:kept], len(items) - kept


def serialize_facts(facts: Iterable[Fact], max_bytes: int = 4096) -> str:
    """Serialize facts up to the UTF-8 byte budget, ending in an explicit
    ``TRUNCATED`` line when anything was omitted."""
    kept, dropped = truncate_facts(facts, max_bytes)
    lines = [format_fact(fact) for fact in kept]
    if dropped:
        lines.append(truncated_line(dropped, len(kept), max_bytes))
    return "\n".join(lines)