# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Official renderers operate on validated registered inputs and private output paths."""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from profile_db.errors import QueryError
from profile_db.runtime_adapter.capture import read_capture, validate_capture, _materialize_chip_spelling
from profile_db.runtime_adapter import Calls, module
from profile_db.runtime_adapter.modalities import events
from profile_db.runtime_adapter.storage import read_snapshot, resolve

RECORD_KINDS = ("chip_swimlane_records", "l2_perf_records", "l2_swimlane_records")


def records_path(conn, run_id):
    kinds = conn.execute("SELECT kind FROM artifact WHERE run_id=?", [run_id]).fetchall()
    kind = next((k[0] for k in kinds if k[0] in RECORD_KINDS), None)
    if kind is None:
        raise QueryError("no records artifact")
    return resolve(conn, run_id, kind)


def export(conn, run_id, kind, fmt, target, *, kernel=None, run_ids=None):
    calls = Calls()
    if kind == "merged":
        if fmt != "perfetto":
            raise QueryError("merged rendering requires perfetto")
        return merged(conn, run_ids or [run_id], target, calls)
    if kind == "deps" and fmt == "html":
        deps = resolve(conn, run_id, "deps")
        edges, nodes, annotations, tensors, tasks = calls.call("deps_viewer", "_load_deps_edges", deps)
        meta = calls.call("deps_viewer", "_load_task_meta", deps)
        target.write_text(
            calls.compatible_call(
                "deps_viewer",
                "emit_html",
                edges,
                nodes,
                meta,
                annotations=annotations,
                tensor_table=tensors,
                task_table=tasks,
            )
        )
    elif kind == "scope" and fmt == "html":
        with TemporaryDirectory(prefix="pfdb-scope-") as work:
            files = calls.call(
                "scope_stats_plot", "process", resolve(conn, run_id, "scope_stats"), Path(work)
            )
            if not files:
                raise QueryError("scope renderer has no paired resources")
            target.write_bytes(Path(files[0]).read_bytes())
    elif kind == "host" and fmt == "perfetto":
        payload = read_snapshot(conn, run_id, "host")
        if payload["status"] != "ok":
            raise QueryError(payload.get("reason", "host analysis unavailable"))
        # Verify linked inputs before exporting the persisted host tree.
        for (key,) in conn.execute(
            "SELECT kind FROM artifact WHERE run_id=? AND kind LIKE 'host_%'", [run_id]
        ).fetchall():
            resolve(conn, run_id, key)
        span_cls = module("strace_timing").Span
        spans = [span_cls(**s) for s in payload["payload"]["spans"]]
        doc = calls.call("strace_timing", "to_host_swimlane", spans)
        target.write_text(json.dumps(doc))
    elif kind == "incore" and fmt == "perfetto":
        if not kernel:
            raise QueryError("incore rendering requires kernel")
        target.write_text(json.dumps(events(resolve(conn, run_id, "incore_trace:" + kernel), calls)))
    elif kind in ("whole", "window", "task", "core") and fmt == "perfetto":
        records = records_path(conn, run_id)
        data = read_capture(records, calls=calls)
        deps = resolve(conn, run_id, "deps")
        name_path = resolve(conn, run_id, "name_map")
        if name_path.suffix == '.py':
            from profile_db.runtime_adapter.kernel_names import load
            names, orch_name = {int(k):v for k,v in load(name_path).items()}, None
        else:
            names, orch_name = calls.call("swimlane_converter", "load_func_names_json", name_path)
        edge_map = calls.call("swimlane_converter", "load_deps_json", deps)
        doc = calls.compatible_call(
            "swimlane_converter",
            "generate_chrome_trace_json",
            data["tasks"],
            None,
            names,
            scheduler_phases=data.get("aicpu_scheduler_phases"),
            scheduler_streams=data.get("scheduler_streams"),
            orchestrator_phases=data.get("aicpu_orchestrator_phases"),
            core_to_thread=data.get("core_to_thread"),
            orchestrator_name=orch_name,
            orchestrator_source=data.get("orchestrator_source"),
            timeline_metadata=data.get("timeline_metadata"),
            deps_edges=edge_map,
            deps_kernel_map=calls.call("swimlane_converter", "load_deps_kernel_map", deps),
            deps_block_map=calls.call("swimlane_converter", "load_deps_block_map", deps),
            emit_overhead=True,
            host_device_uploads=data.get("host_device_uploads"),
            aicpu_lifecycle_records=data.get("aicpu_lifecycle_records"),
        )
        target.write_text(json.dumps(doc))
    else:
        raise QueryError(f"unsupported render combination: {kind}/{fmt}")
    return dict(functions=calls.functions, compatibility=calls.compatibility)


def merged(conn, run_ids, target, calls):
    if len(run_ids) < 2 or len(run_ids) != len(set(run_ids)):
        raise QueryError("merged rendering requires at least two distinct run_ids")
    domains = set()
    with TemporaryDirectory(prefix="pfdb-merge-") as work:
        root = Path(work)
        host_files = {}
        ranks = set()
        identities = []
        for run_id in run_ids:
            row = conn.execute("SELECT rank_label FROM run WHERE run_id=?", [run_id]).fetchone()
            match = re.fullmatch(r"(?:rank)?(\d+)", row[0] if row else "")
            if not match or match[1] in ranks:
                raise QueryError("merged runs require distinct numeric rank labels")
            ranks.add(match[1])
            records = records_path(conn, run_id)
            identity = None
            if conn.execute(
                "SELECT 1 FROM artifact WHERE run_id=? AND kind='dispatch_identity'", [run_id]
            ).fetchone():
                identity = json.loads(resolve(conn, run_id, "dispatch_identity").read_text())
            identities.append(identity)
            index = identity["local_capture_index"] if identity else 0
            dest = root / ("rank" + match[1]) / f"d{index}"
            dest.mkdir(parents=True)
            raw = json.loads(records.read_text())
            domain = raw.get("metadata", {}).get("host_clock_domain_id")
            if not domain:
                raise QueryError("merged captures require host_clock_domain_id")
            domains.add(domain)
            validate_capture(raw)
            projected, temporary = _materialize_chip_spelling(records, raw)
            try:
                (dest / "chip_swimlane_records.json").write_bytes(projected.read_bytes())
                if temporary:
                    calls.compatibility.append("capture-single-epoch-v1")
            finally:
                if temporary:
                    projected.unlink(missing_ok=True)
            for kind, name in (("deps", "deps.json"), ("name_map", "name_map.json")):
                path = resolve(conn, run_id, kind)
                if kind == 'name_map' and path.suffix == '.py':
                    from profile_db.runtime_adapter.kernel_names import document
                    (dest / name).write_text(json.dumps(document(path)))
                else:
                    (dest / name).write_bytes(path.read_bytes())
            for (kind,) in conn.execute("SELECT kind FROM artifact WHERE run_id=?", [run_id]).fetchall():
                if kind == "dispatch_identity":
                    (dest / "dispatch_identity.json").write_bytes(resolve(conn, run_id, kind).read_bytes())
                elif kind.startswith("host_log:"):
                    path = resolve(conn, run_id, kind)
                    content = path.read_bytes()
                    if path.name in host_files and host_files[path.name] != content:
                        raise QueryError("conflicting host logs")
                    host_files[path.name] = content
        if len(domains) != 1:
            raise QueryError("captures belong to different host clocks")
        if not host_files:
            raise QueryError("merged rendering requires attached host logs")
        for name, content in host_files.items():
            (root / name).write_bytes(content)
        parents = {(i["run_id"], i["task_slot"]) for i in identities if i}
        if parents and (len(parents) != 1 or any(i is None for i in identities)):
            raise QueryError("merged captures require matching dispatch identities on all ranks")
        parent = next(iter(parents)) if parents else None
        selector = f"{parent[0]}:{parent[1]}" if parent and identities[0]["group_size"] > 1 else None
        args = SimpleNamespace(
            func_names=None,
            kernel_config=None,
            deps_json=None,
            dispatch=None if selector else f"d{index}",
            dispatch_id=selector,
            host_log=None,
            rank_pid=[],
            verbose=False,
            overhead=True,
            output=str(target),
        )
        calls.call("swimlane_converter", "_generate_l3_trace", args, root)
        doc = json.loads(target.read_text())
        doc["metadata"]["host_logs"] = sorted(host_files)
        for rank in doc["metadata"]["ranks"]:
            rank["input"] = str(Path(rank["input"]).relative_to(root))
        target.write_text(json.dumps(doc))
    return dict(functions=calls.functions, compatibility=calls.compatibility)
