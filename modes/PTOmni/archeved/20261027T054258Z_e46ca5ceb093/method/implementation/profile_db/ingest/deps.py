# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""deps.json parsing: logical tasks and dependency edges, fields preserved verbatim.

Every raw field is preserved (DESIGN.md 5.2/5.3). Kernel names resolve
through the name_map: the non-negative entries of a task's ``kernel_ids``
map to kernel names, and the family strips the ``_<n>_aic/aiv`` or
``_<n>`` suffix convention.

Edge validation reflects the runtime's graph semantics (verified on a
real Qwen3Decode capture): ``succ`` must always be a real task, while
``pred`` may refer to an external creator for ``source="creator"``
edges, which enter the graph from a host-side pseudo node.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profile_db.errors import IngestError
from profile_db.task_ids import normalize_task_id

_FAMILY_SUFFIX = re.compile(r"(?:_\d+)?_(?:aic|aiv)$|(?:_\d+)$")


def family_of(name: str) -> str:
    """Strip the per-engine/tile numbering suffix from a kernel name."""
    return _FAMILY_SUFFIX.sub("", name)


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


@dataclass
class TaskInfo:
    """One logical task with resolved names; timing columns stay None until
    the swimlane join fills them (ingest orchestrator)."""

    task_id: str
    task_id_raw: str
    scope: str
    early_dispatch: bool
    kernel_ids: list[int]
    block_num: int
    name: str | None
    family: str | None
    # filled from the swimlane:
    engine: str | None = None
    num_rows: int = 0
    busy_us: float | None = None
    wall_us: float | None = None
    min_dispatch_us: float | None = None
    min_receive_us: float | None = None
    min_start_us: float | None = None
    max_end_us: float | None = None
    max_finish_us: float | None = None


@dataclass
class DepGraph:
    """Parsed dependency graph with task/edge rows ready for the writer."""

    tasks: list[TaskInfo] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    task_ids: set[str] = field(default_factory=set)

    def task(self, task_id: str) -> TaskInfo:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{path}: malformed JSON: {exc}") from exc
    except OSError as exc:
        raise IngestError(f"{path}: cannot read: {exc}") from exc
    if not isinstance(data, dict):
        raise IngestError(f"{path}: top-level JSON must be an object")
    return data


def load_deps(deps_path: Path, name_map: dict[str, str]) -> DepGraph:
    """Parse deps.json: every raw field preserved, names resolved through
    the name_map, structural violations reported as IngestError."""
    data = _read_json(deps_path)
    tasks = data.get("tasks")
    edges = data.get("edges")
    if not isinstance(tasks, list) or not tasks:
        raise IngestError(f"{deps_path}: 'tasks' must be a non-empty list")
    # A standalone task, or several independent tasks, legitimately has no
    # dependency edges. Missing/non-list edges still indicate malformed input.
    if not isinstance(edges, list):
        raise IngestError(f"{deps_path}: 'edges' must be a list")

    graph = DepGraph()
    for index, raw in enumerate(tasks):
        if "task_id" not in raw:
            raise IngestError(f"{deps_path}: task {index} has no task_id")
        task_identity = normalize_task_id(raw["task_id"])
        task_id = task_identity.canonical
        if task_id in graph.task_ids:
            raise IngestError(f"{deps_path}: duplicate task_id {task_id}")
        graph.task_ids.add(task_id)
        kernel_ids = [int(v) for v in (raw.get("kernel_ids") or [])]
        callables = [v for v in kernel_ids if v >= 0]
        names = _dedupe(
            [name_map.get(str(v), "") for v in callables if str(v) in name_map]
        )
        name = "+".join(names) if names else None
        families = _dedupe([family_of(n) for n in names if n])
        family = "+".join(families) if families else None
        graph.tasks.append(
            TaskInfo(
                task_id=task_id,
                task_id_raw=task_identity.raw,
                scope=str(raw.get("scope") or ""),
                early_dispatch=bool(raw.get("early_dispatch")),
                kernel_ids=kernel_ids,
                block_num=int(raw.get("block_num") or 0),
                name=name,
                family=family,
            )
        )

    for index, raw in enumerate(edges):
        for key in ("pred", "succ"):
            if key not in raw:
                raise IngestError(f"{deps_path}: edge {index} missing {key!r}")
        succ_identity = normalize_task_id(raw["succ"])
        succ = succ_identity.canonical
        if succ not in graph.task_ids:
            raise IngestError(f"{deps_path}: edge {index} succ {succ} is not a known task")
        pred_identity = normalize_task_id(raw["pred"])
        pred = pred_identity.canonical
        if raw.get("source") != "creator" and pred not in graph.task_ids:
            raise IngestError(
                f"{deps_path}: edge {index} pred {pred} is not a known task "
                "and source is not creator"
            )
        flags = raw.get("flags") or []
        if not isinstance(flags, list):
            raise IngestError(f"{deps_path}: edge {index} 'flags' must be a list")
        graph.edges.append(
            {
                "pred": pred,
                "succ": succ,
                "pred_raw": pred_identity.raw,
                "succ_raw": succ_identity.raw,
                "source": str(raw.get("source") or ""),
                "arg": str(raw.get("arg") or ""),
                "flags": flags,
                "tensor_id": str(raw.get("tensor_id") or ""),
                "consumer_dtype": str(raw.get("consumer_dtype") or ""),
                "consumer_shape": raw.get("consumer_shape") or [],
                "consumer_start_offset": str(raw.get("consumer_start_offset") or ""),
                "consumer_strides": raw.get("consumer_strides") or [],
            }
        )
    return graph
