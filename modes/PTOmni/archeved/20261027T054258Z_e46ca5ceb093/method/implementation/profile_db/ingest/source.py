# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Capture source discovery and artifact naming normalization.

A capture source is the ``dfx_outputs/`` directory of one profiled run.
The records file has drifted across runtime generations
(``chip_swimlane_records.json`` is current, ``l2_swimlane_records.json``
is the documented name, ``l2_perf_records.json`` is the readable legacy
name); all three are accepted and normalized to one ingest path.
Artifact paths are recorded relative to the source's parent so no
machine-specific absolute path ever enters the database.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profile_db.errors import IngestError

RECORD_NAMES = (
    "chip_swimlane_records.json",
    "l2_swimlane_records.json",
    "l2_perf_records.json",
)

_NAME_MAP_PATTERN = re.compile(r"^name_map_(?P<program>.+?)(?:_(?P<ts>\d{8}_\d{6}))?\.json$")
_DIR_TS_PATTERN = re.compile(r"_(20\d{6}_\d{6})$")

_LEVEL_KEYS = ("chip_swimlane_level", "l2_swimlane_level")


@dataclass(frozen=True)
class Source:
    """One discovered capture directory with every required artifact."""

    path: Path                    # the dfx_outputs directory
    records: Path                 # the records file (any accepted name)
    records_kind: str             # normalized artifact kind
    deps: Path
    name_map: Path
    merged: Path | None           # optional on simulator platforms
    program: str | None           # parsed from name_map_<Program>_<ts>.json
    captured_at: str | None       # parsed from the name_map timestamp


def _one_json(directory: Path, names: tuple[str, ...], what: str) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise IngestError(f"capture {directory} is missing {what} ({', '.join(names)})")


def discover_source(path: Path | str) -> Source:
    """Validate a capture directory and resolve every artifact (raises
    ``IngestError`` with a precise message when anything is missing)."""
    directory = Path(path)
    if not directory.is_dir():
        raise IngestError(f"capture directory does not exist: {directory}")
    records = _one_json(directory, RECORD_NAMES, "the swimlane records file")
    deps = _one_json(directory, ("deps.json",), "deps.json")
    name_maps = list(directory.glob("name_map_*.json"))
    dispatch_program = None
    if not name_maps and (directory/'dispatch_program.json').is_file():
        dispatch_program = json.loads((directory/'dispatch_program.json').read_text()).get('program')
        if not isinstance(dispatch_program,str) or not re.fullmatch(r'[A-Za-z_]\w*',dispatch_program):
            raise IngestError('invalid dispatch program identity')
        # Current distributed layout is build/dfx_outputs/rankR/dD. Only
        # this fixed ancestry may select the compiled child configuration.
        if (not re.fullmatch(r'd\d+',directory.name) or not re.fullmatch(r'rank\d+',directory.parent.name)
                or directory.parent.parent.name != 'dfx_outputs'):
            raise IngestError('unsupported dispatch capture layout')
        config = directory.parent.parent.parent/'next_levels'/dispatch_program/'kernel_config.py'
        if not config.is_file() or config.is_symlink():
            raise IngestError('dispatch kernel configuration missing')
        name_maps = [config]
    if len(name_maps) != 1:
        raise IngestError(
            f"capture {directory} must contain exactly one name_map_*.json (found {len(name_maps)})"
        )
    name_map = name_maps[0]
    # Current runners name the file after the work dir (no timestamp);
    # older ones appended _YYYYMMDD_HHMMSS. The program is everything
    # after the prefix in both spellings, and the timestamp stays None
    # when absent.
    match = _NAME_MAP_PATTERN.match(name_map.name)
    program = match.group("program") if match else dispatch_program
    captured_at = match.group("ts") if match else None
    return Source(
        path=directory,
        records=records,
        records_kind=_records_kind(records.name),
        deps=deps,
        name_map=name_map,
        merged=_merged(directory),
        program=program,
        captured_at=captured_at,
    )


def _records_kind(filename: str) -> str:
    for name in RECORD_NAMES:
        if filename == name:
            return name.removesuffix(".json")
    raise IngestError(f"unexpected records filename: {filename}")


def _merged(directory: Path) -> Path | None:
    candidates = list(directory.glob("merged_swimlane_*.json"))
    if len(candidates) > 1:
        raise IngestError(f"capture {directory} has {len(candidates)} merged traces; expected at most one")
    return candidates[0] if candidates else None


def rel_path(source: Source, artifact: Path) -> str:
    """Repository-relative path of an artifact (relative to the capture's
    parent when possible; bare filename otherwise)."""
    base = source.path.resolve().parent
    resolved = artifact.resolve()
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return resolved.name


def records_level(data: dict[str, Any]) -> int:
    """The sampled level from either accepted top-level key of an already
    parsed records document."""
    for key in _LEVEL_KEYS:
        if key in data:
            level = data[key]
            if not isinstance(level, int) or not 1 <= level <= 4:
                raise IngestError(
                    f"swimlane records: {key} must be an int in 1..4, got {level!r}"
                )
            return level
    raise IngestError(
        f"swimlane records: missing level key (expected one of {_LEVEL_KEYS})"
    )


def load_name_map(name_map: Path) -> dict[str, str]:
    """callable_id -> kernel name from the name_map artifact."""
    from profile_db.runtime_adapter.kernel_names import load
    return load(name_map)


def raw_records(records: Path) -> dict[str, Any]:
    """Parse the records file, wrapping JSON damage into IngestError."""
    try:
        data = json.loads(records.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{records}: malformed JSON: {exc}") from exc
    except OSError as exc:
        raise IngestError(f"{records}: cannot read: {exc}") from exc
    if not isinstance(data, dict):
        raise IngestError(f"{records}: top-level JSON must be an object")
    return data
