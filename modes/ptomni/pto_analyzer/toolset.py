from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from typing import Any

from .diagnostics import compute, dataflow, schedule
from .diagnostics.common import PAGE_SIZE_DEFAULT, PAGE_SIZE_MAX
from .profile import ProfileArchive, ProfileRef

_TOOLS = {
    "Compute": {
        "compute_mapping_check": compute.compute_mapping_check,
        "compute_unit_activity": compute.compute_unit_activity,
        "compute_task_share": compute.compute_task_share,
    },
    "Dataflow": {
        "tile_access_check": dataflow.tile_access_check,
        "buffer_pressure_check": dataflow.buffer_pressure_check,
    },
    "Schedule": {
        "dependency_path_check": schedule.dependency_path_check,
        "dispatch_gap_check": schedule.dispatch_gap_check,
        "task_distribution_check": schedule.task_distribution_check,
    },
}
_DESCRIPTIONS = {
    "compute_mapping_check": "Summarize archived program and callable mappings without unverified Candidate source attribution.",
    "compute_unit_activity": "Aggregate all PMU counters by archived build, rank, dispatch, and core type.",
    "compute_task_share": "Aggregate PMU counters by task only after same-dispatch task and callable joins validate.",
    "tile_access_check": "Summarize all archived compiler hint categories and preserve their original text.",
    "buffer_pressure_check": "Summarize MemoryReuse hints and Scope fields without inferring capacity pressure.",
    "dependency_path_check": "Summarize normalized dependency graph structure without runtime weights.",
    "dispatch_gap_check": "Summarize same-core AICore interval gaps in cycles; these gaps are not attributed to scheduling.",
    "task_distribution_check": "Summarize AICore interval durations per core, tuple counts, and distinct raw task tokens without inferring busy time or load balance.",
}
_TUNER_BY_NAME = {name: tuner for tuner, tools in _TOOLS.items() for name in tools}
_CURSOR_VERSION = 2
_CURSOR_FIELDS = {"after", "candidate_sha256", "event_id", "tool", "tuner", "version"}
_CURSOR_PATTERN = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\Z")
_CURSOR_SECRET = secrets.token_bytes(32)
_CURSOR_KEY_TYPES = {
    "compute_mapping_check": (str, str, str),
    "compute_unit_activity": (str, str, str, str, int),
    "compute_task_share": (str, str, str, str, str),
    "tile_access_check": (str, int, str, str),
    "buffer_pressure_check": (str, int, str),
    "dependency_path_check": (str, str, str, str, str),
    "dispatch_gap_check": (str, str, int),
    "task_distribution_check": (str, str, int),
}
_PAGINATED_TOOLS = frozenset(_CURSOR_KEY_TYPES)


@dataclass(frozen=True)
class ToolSession:
    """Read-only tool interface bound to one Tuner and one profile event."""

    tuner: str
    _archive: ProfileArchive

    @property
    def profile(self) -> dict[str, Any]:
        return self._archive.identity.copy()

    @property
    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": (
                    _DESCRIPTIONS[name]
                    + " Returns details with total_records, returned_records, records, and next_cursor (null on the last page)."
                    if name in _PAGINATED_TOOLS else _DESCRIPTIONS[name]
                ),
                "input_schema": {
                    "type": "object",
                    "properties": ({
                        "cursor": {"type": "string", "description": "Opaque continuation token returned by this tool."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": PAGE_SIZE_MAX,
                                  "description": f"Maximum detail records to return; defaults to {PAGE_SIZE_DEFAULT}."},
                    } if name in _PAGINATED_TOOLS else {}),
                    "additionalProperties": False,
                },
            }
            for name in _TOOLS[self.tuner]
        ]

    @property
    def initial_summary(self) -> list[dict[str, Any]]:
        results = []
        for name in _TOOLS[self.tuner]:
            diagnostic = _TOOLS[self.tuner][name]
            result = diagnostic(self._archive, include_details=False) if name in _PAGINATED_TOOLS else diagnostic(self._archive)
            results.append(result)
        return results

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in _TUNER_BY_NAME:
            raise KeyError(f"unknown PTOAnalyzer tool: {name}")
        if _TUNER_BY_NAME[name] != self.tuner:
            raise PermissionError(f"{self.tuner} Tuner cannot call {name}")
        if name in _PAGINATED_TOOLS:
            if args is None:
                args = {}
            if not isinstance(args, dict) or set(args) - {"cursor", "limit"}:
                raise ValueError(f"{name} accepts only cursor and limit")
            cursor = args.get("cursor")
            after = _decode_cursor(cursor, name, self.tuner, self._archive.identity) if cursor is not None else None
            limit = args.get("limit", PAGE_SIZE_DEFAULT)
            if "cursor" in args and type(cursor) is not str:
                raise ValueError("cursor must be a string")
            if type(limit) is not int or not 1 <= limit <= PAGE_SIZE_MAX:
                raise ValueError(f"limit must be an integer from 1 to {PAGE_SIZE_MAX}")
        elif args not in (None, {}):
            raise ValueError(f"{name} does not accept arguments")

        diagnostic = _TOOLS[self.tuner][name]
        if name in _PAGINATED_TOOLS:
            result = diagnostic(self._archive, detail_cursor=after, detail_limit=limit)
        else:
            result = diagnostic(self._archive)
        detail_records = result.pop("_detail_records", None)
        if name in _PAGINATED_TOOLS:
            if detail_records is None:
                raise ValueError(f"{name} has no paginated detail records")
            last_key = result.pop("_detail_last_key", None)
            has_more = result.pop("_detail_has_more", False)
            result["details"].update({
                "returned_records": len(detail_records),
                "next_cursor": _encode_cursor(name, self.tuner, self._archive.identity, last_key) if has_more else None,
                "records": detail_records,
            })
        return result


def _encode_cursor(tool: str, tuner: str, profile: dict[str, Any], after: tuple[Any, ...] | None) -> str:
    if after is None:
        raise ValueError("cannot encode a cursor without a continuation key")
    payload = {
        "version": _CURSOR_VERSION,
        "tool": tool,
        "tuner": tuner,
        "event_id": profile["event_id"],
        "candidate_sha256": profile["candidate_sha256"],
        "after": [list(after[0]), after[1]],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body = base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")
    signature = hmac.new(_CURSOR_SECRET, body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{base64.urlsafe_b64encode(signature).decode('ascii').rstrip('=')}"


def _decode_cursor(cursor: Any, tool: str, tuner: str, profile: dict[str, Any]) -> tuple[Any, ...]:
    if not isinstance(cursor, str) or len(cursor) > 8192 or not _CURSOR_PATTERN.fullmatch(cursor):
        raise ValueError("invalid cursor")
    body, signature = cursor.split(".")
    try:
        raw = _decode_cursor_part(body)
        supplied_signature = _decode_cursor_part(signature)
        expected_signature = hmac.new(_CURSOR_SECRET, body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError("invalid cursor signature")
        payload = json.loads(raw)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid cursor") from exc
    if not isinstance(payload, dict) or set(payload) != _CURSOR_FIELDS:
        raise ValueError("invalid cursor")
    if type(payload["version"]) is not int or payload["version"] != _CURSOR_VERSION:
        raise ValueError("unsupported cursor version")
    if (
        payload["tool"] != tool
        or payload["tuner"] != tuner
        or payload["event_id"] != profile["event_id"]
        or payload["candidate_sha256"] != profile["candidate_sha256"]
    ):
        raise ValueError("cursor does not match this ToolSession")
    after = payload["after"]
    if (
        not isinstance(after, list)
        or len(after) != 2
        or not isinstance(after[0], list)
        or len(after[0]) != len(_CURSOR_KEY_TYPES[tool])
        or any(type(value) is not expected for value, expected in zip(after[0], _CURSOR_KEY_TYPES[tool]))
        or type(after[1]) is not int
        or after[1] < 1
    ):
        raise ValueError("invalid cursor")
    return tuple(after[0]), after[1]


def _decode_cursor_part(value: str) -> bytes:
    raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    canonical = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    if canonical != value:
        raise ValueError("non-canonical cursor encoding")
    return raw


def open_toolset(tuner: str, profile_ref: ProfileRef) -> ToolSession:
    """Open one authorized toolset for a caller-selected, validated event."""
    if tuner not in _TOOLS:
        raise ValueError(f"unsupported Tuner: {tuner}")
    if not isinstance(profile_ref, ProfileRef):
        raise TypeError("profile_ref must be a PTOAnalyzer ProfileRef")
    return ToolSession(tuner, ProfileArchive(profile_ref))
