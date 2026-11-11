# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The query registry (DESIGN.md 6.6).

Every query registers here with three mandatory pieces: a stable name, an
**owner question** (the kernel-agent information need it answers — no
question, no query), and its pydantic parameter model (the single source
the CLI/MCP entries are generated from). Handlers are pure ``(conn, params)
-> list[Fact]`` functions that read only schema tables and the derived
layer's tested pure functions.

``execute`` resolves a name, validates parameters, runs the handler, and
renders the facts under a byte budget. Unknown names and invalid
parameters raise ``QueryError`` (structured, never a traceback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from pydantic import BaseModel, ValidationError

from profile_db.errors import PfdbError, QueryError
from profile_db.facts import Fact
from profile_db.query.params import (
    CoreParams,
    CriticalPathParams,
    DensityParams,
    DepsParams,
    EarlyDispatchParams,
    InventoryParams,
    MemoryParams,
    OverviewParams,
    PerfHintsParams,
    PmuParams,
    RegionParams,
    RowsParams,
    RunsListParams,
    SchedulerParams,
    SparseRegionsParams,
    SubgraphParams,
    TaskParams,
    WhyLateParams,
    WhyLongParams,
    WhySparseParams,
)

QueryHandler = Callable[..., list[Fact]]


@dataclass(frozen=True)
class QuerySpec:
    """One registered query: identity, rationale, parameters, handler."""

    name: str
    owner_question: str
    params: type[BaseModel]
    handler: QueryHandler
    rank_axis: bool = False
    default_budget_bytes: int = 4096


_REGISTRY: dict[str, QuerySpec] = {}
_ORDER: list[str] = []


def register(
    name: str,
    owner_question: str,
    params: type[BaseModel],
    *,
    rank_axis: bool = False,
    default_budget_bytes: int = 4096,
) -> Callable[[QueryHandler], QueryHandler]:
    """Decorator: attach a handler as a named, questioned query."""
    if not name or not owner_question:
        raise QueryError("query registration requires a name and an owner question")
    if name in _REGISTRY:
        raise QueryError(f"query {name!r} is registered twice")
    if default_budget_bytes < 1:
        raise QueryError("default_budget_bytes must be at least 1")

    def decorator(handler: QueryHandler) -> QueryHandler:
        _REGISTRY[name] = QuerySpec(
            name=name,
            owner_question=owner_question,
            params=params,
            handler=handler,
            rank_axis=rank_axis,
            default_budget_bytes=default_budget_bytes,
        )
        _ORDER.append(name)
        return handler

    return decorator


def list_queries() -> Sequence[QuerySpec]:
    """Registered queries in registration order."""
    return tuple(_REGISTRY[name] for name in _ORDER)


def get_query(name: str) -> QuerySpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(_ORDER)
        raise QueryError(f"unknown query {name!r}; available: {known}") from exc


def _coerce(params: QuerySpec, raw: Mapping[str, object] | BaseModel | None) -> BaseModel:
    if raw is None:
        raw = {}
    if isinstance(raw, BaseModel):
        return raw
    if not isinstance(raw, Mapping):
        raise QueryError(f"query {params.name!r} parameters must be a mapping")
    try:
        return params.params.model_validate(dict(raw))
    except ValidationError as exc:
        raise QueryError(f"invalid parameters for {params.name!r}: {exc}") from exc


def execute(
    conn,
    name: str,
    params: Mapping[str, object] | BaseModel | None = None,
    *,
    budget_bytes: int | None = None,
):
    """Run one query and render its facts under the byte budget.

    ``budget_bytes=None`` uses the query's registered default (4096 unless
    the query opted into a larger one). Any non-``PfdbError`` escaping a
    handler is a bug in that handler, but it must still reach the caller
    as a structured error: CLI and MCP both catch ``PfdbError`` only, so a
    bare exception would surface as a traceback on the agent's channel.
    """
    from profile_db.query.result import render

    spec = get_query(name)
    model = _coerce(spec, params)
    _validate_rank(conn, model)
    try:
        facts = spec.handler(conn, model)
        if hasattr(model, "run_id"):
            from profile_db.runtime_adapter.storage import read_snapshot
            from profile_db.facts import Evidence

            doc = read_snapshot(conn, model.run_id, "critical_path")
            if doc["status"] != "ok":
                unavailable = Fact(
                    "ANALYSIS",
                    dict(
                        run_id=model.run_id,
                        family="critical_path",
                        status=doc["status"],
                        reason=doc.get("reason", "analysis unavailable"),
                    ),
                    Evidence.UNAVAILABLE,
                )
                if (
                    name == "critical_path"
                    or (name == "why_sparse" and doc["status"] == "stale")
                    or (name == "idle_window" and model.until_task_id is None)
                    or (name == "tasks" and model.on_cpm is not None)
                ):
                    facts = [unavailable]
                else:
                    keys = {"cpm_us", "on_cpm_observed", "on_cpm_static"}
                    if any(keys.intersection(f.fields) for f in facts):
                        facts = [
                            Fact(f.rec, {k: v for k, v in f.fields.items() if k not in keys}, f.evidence)
                            for f in facts
                        ]
                        facts.append(unavailable)

    except PfdbError:
        raise
    except Exception as exc:
        raise QueryError(f"query {name!r} failed: {type(exc).__name__}: {exc}") from exc
    resolved = spec.default_budget_bytes if budget_bytes is None else budget_bytes
    return render(facts, resolved)


def _validate_rank(conn, model: BaseModel) -> None:
    """Check an optional rank selector against a run-scoped request.

    A run id already identifies one capture, so unqualified run queries are
    deterministic even in a multi-rank database. Supplying ``rank`` remains
    useful as an explicit guard in scripts: a typo must not silently query a
    similarly numbered run from another rank.
    """
    if not hasattr(model, "run_id") or getattr(model, "rank", None) is None:
        return
    row = conn.execute("SELECT rank_label FROM run WHERE run_id = ?", [getattr(model, "run_id")]).fetchone()
    if row is not None and row[0] != getattr(model, "rank"):
        raise QueryError(
            f"run {getattr(model, 'run_id')} belongs to rank {row[0]!r}, "
            f"not requested rank {getattr(model, 'rank')!r}"
        )


__all__ = [
    "register",
    "execute",
    "get_query",
    "list_queries",
    "QuerySpec",
    "CoreParams",
    "CriticalPathParams",
    "DensityParams",
    "DepsParams",
    "EarlyDispatchParams",
    "InventoryParams",
    "MemoryParams",
    "OverviewParams",
    "PerfHintsParams",
    "PmuParams",
    "RegionParams",
    "RowsParams",
    "RunsListParams",
    "SchedulerParams",
    "SparseRegionsParams",
    "SubgraphParams",
    "TaskParams",
    "WhyLateParams",
    "WhyLongParams",
    "WhySparseParams",
]
