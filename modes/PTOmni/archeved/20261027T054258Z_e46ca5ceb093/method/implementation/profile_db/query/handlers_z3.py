# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Z3 handlers: a single operator's identity, timing, and dependency
neighborhood."""

from __future__ import annotations

import collections
from typing import Any

from profile_db.errors import QueryError
from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.registry import register
from profile_db.query.params import DepsParams, SubgraphParams, TaskParams, TasksParams

_DEP_COLS = (
    "pred, succ, source, arg, CAST(flags AS VARCHAR), tensor_id, consumer_dtype, "
    "CAST(consumer_shape AS VARCHAR), consumer_start_offset, CAST(consumer_strides AS VARCHAR)"
)


@register(
    "task",
    "Locate/constraints: what is this operator's identity, timing, and "
    "critical-path membership?",
    TaskParams,
)
def task_detail(conn, params: TaskParams) -> list[Fact]:
    fact = common.task_fact(conn, params.run_id, params.task_id)
    if fact is None:
        return [
            Fact(
                "TASK",
                common.fields(run_id=params.run_id, task_id=params.task_id),
                Evidence.UNAVAILABLE,
            )
        ]
    return [fact]


@register(
    "tasks",
    "Locate: which task_ids belong to this family or name so the next "
    "task/pmu/why_long call does not walk the critical path hop by hop?",
    TasksParams,
    default_budget_bytes=16384,
)
def tasks(conn, params: TasksParams) -> list[Fact]:
    if params.family is None and params.name is None:
        raise QueryError("tasks requires --family or --name (refusing to dump the whole graph)")
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("TASK", run_id)
    sql = "SELECT task_id FROM task WHERE run_id = ?"
    args: list[Any] = [run_id]
    if params.family is not None:
        sql += " AND family = ?"
        args.append(params.family)
    if params.name is not None:
        sql += " AND instr(name, ?) > 0"
        args.append(params.name)
    if params.engine is not None:
        sql += " AND engine = ?"
        args.append(params.engine)
    if params.on_cpm == "observed":
        sql += " AND on_cpm_observed = TRUE"
    elif params.on_cpm == "static":
        sql += " AND on_cpm_static = TRUE"
    rows = common.q(conn, sql, args)
    task_ids = sorted((str(row[0]) for row in rows), key=common.num_key)
    if not task_ids:
        return [
            Fact(
                "TASK",
                common.fields(
                    run_id=run_id,
                    family=params.family,
                    name=params.name,
                    engine=params.engine,
                    on_cpm=params.on_cpm,
                ),
                Evidence.UNAVAILABLE,
            )
        ]
    return common.task_facts(conn, run_id, task_ids)


@register(
    "deps",
    "Constraints: who are this operator's direct dependencies, and what tensor "
    "shape/stride/dtype rides each edge?",
    DepsParams,
)
def deps(conn, params: DepsParams) -> list[Fact]:
    from profile_db.query import runtime

    answer = runtime.deps(conn, params)
    if answer is not None:
        return answer
    run_id = params.run_id
    if common.task_fact(conn, run_id, params.task_id) is None:
        return [
            Fact(
                "DEP",
                common.fields(run_id=run_id, task_id=params.task_id, direction=params.direction),
                Evidence.UNAVAILABLE,
            )
        ]
    sql = f"SELECT {_DEP_COLS} FROM dep_edge WHERE run_id = ?"
    args: list[Any] = [run_id]
    if params.direction == "out":
        # outgoing: edges whose pred is this task (its consumers)
        sql += " AND pred = ?"
        args.append(params.task_id)
    elif params.direction == "in":
        # incoming: edges whose succ is this task (its producers)
        sql += " AND succ = ?"
        args.append(params.task_id)
    else:
        sql += " AND (succ = ? OR pred = ?)"
        args.extend([params.task_id, params.task_id])
    rows = common.q(conn, sql + " ORDER BY edge_id", args)
    if not rows:
        return [
            Fact(
                "DEP",
                common.fields(run_id=run_id, task_id=params.task_id, direction=params.direction),
                Evidence.UNAVAILABLE,
            )
        ]
    return [common.dep_fact(run_id, row) for row in rows]


@register(
    "subgraph",
    "Constraints: how deep and how wide is this operator's in/out "
    "neighborhood?",
    SubgraphParams,
)
def subgraph(conn, params: SubgraphParams) -> list[Fact]:
    run_id = params.run_id
    if common.task_fact(conn, run_id, params.task_id) is None:
        return [
            Fact(
                "SUBGRAPH",
                common.fields(run_id=run_id, task_id=params.task_id),
                Evidence.UNAVAILABLE,
            )
        ]
    succ: dict[str, list[str]] = collections.defaultdict(list)
    preds: dict[str, list[str]] = collections.defaultdict(list)
    for pred, succ_id in common.q(
        conn, "SELECT pred, succ FROM dep_edge WHERE run_id = ? ORDER BY edge_id", [run_id]
    ):
        succ[pred].append(succ_id)
        preds[succ_id].append(pred)

    node_depth: dict[str, int] = {params.task_id: 0}
    frontier = [params.task_id]
    depth = 0
    capped = False
    while frontier and depth < params.depth:
        nxt: list[str] = []
        for node in frontier:
            for child in succ.get(node, ()):
                if child in node_depth:
                    continue
                if len(node_depth) >= params.max_nodes:
                    capped = True
                    continue
                node_depth[child] = depth + 1
                nxt.append(child)
        frontier = nxt
        depth += 1
    # include direct producers for context (depth -1), capped the same way
    for pred in preds.get(params.task_id, ()):
        if pred in node_depth:
            continue
        if len(node_depth) >= params.max_nodes:
            capped = True
            continue
        node_depth[pred] = -1

    nodes = sorted(node_depth, key=common.num_key)
    # A ``pred`` may be a host-side creator pseudo node with no task row
    # (deps.json keeps those verbatim); report it as an external node
    # instead of assuming every neighbor is a task.
    identities = common.task_identities(conn, run_id, nodes)
    facts: list[Fact] = [
        Fact(
            "SUBGRAPH",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                depth=params.depth,
                nodes=len(node_depth),
                external_nodes=sum(1 for node in nodes if node not in identities) or None,
                capped=capped,
            ),
            Evidence.MEASURED,
        )
    ]
    for node in nodes:
        identity = identities.get(node)
        if identity is None:
            facts.append(
                Fact(
                    "NODE",
                    common.fields(
                        run_id=run_id,
                        task_id=node,
                        kind="external",
                        depth=node_depth[node],
                    ),
                    Evidence.UNAVAILABLE,
                )
            )
            continue
        name, family, engine = identity
        facts.append(
            Fact(
                "NODE",
                common.fields(
                    run_id=run_id,
                    task_id=node,
                    name=name,
                    family=family,
                    engine=engine,
                    depth=node_depth[node],
                ),
                Evidence.MEASURED,
            )
        )
    in_subgraph = set(node_depth)
    marks = ", ".join("?" for _ in nodes)
    edge_rows = common.q(
        conn,
        f"SELECT {_DEP_COLS} FROM dep_edge WHERE run_id = ? "
        f"AND pred IN ({marks}) ORDER BY edge_id",
        [run_id, *nodes],
    )
    facts.extend(
        common.dep_fact(run_id, row)
        for row in edge_rows
        if row[1] in in_subgraph
    )
    return facts