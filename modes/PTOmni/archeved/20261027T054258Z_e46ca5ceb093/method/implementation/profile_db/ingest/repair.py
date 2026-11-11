# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit, hash-anchored physical rebuild; refresh only updates analyses."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from profile_db.errors import IngestError, QueryError
from profile_db.runtime_adapter import digest
from profile_db.runtime_adapter.storage import resolve
from profile_db.schema import RECORD_KINDS

TABLES = ("task", "task_row", "dep_edge", "scheduler_phase", "orch_phase", "analysis_snapshot")


def counts(conn, run_id):
    return {t: conn.execute(f"SELECT count(*) FROM {t} WHERE run_id=?", [run_id]).fetchone()[0] for t in TABLES}


def validate(context):
    for kind, path, sha in context["inputs"]:
        if not path.is_file() or digest(path) != sha:
            raise IngestError("repair input changed or missing: " + kind)


def restore_artifacts(conn, run_id, context, copied):
    validate(context)
    conn.execute("DELETE FROM artifact WHERE run_id=?", [run_id])
    columns = context["columns"]
    for row in context["artifacts"]:
        conn.execute("INSERT INTO artifact (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")", row)
    # Recover missing archived inputs into the managed store atomically with
    # the rebuild; ingest cleans newly copied files if the transaction fails.
    from profile_db.ingest.analysis import register_paths
    for raw in context["artifacts"]:
        artifact = dict(zip(columns, raw))
        if artifact["store_mode"] == "copy":
            from profile_db.runtime_adapter.storage import base_dir
            if not (base_dir(conn)/artifact["locator"]).is_file():
                path = next(path for kind,path,_ in context["inputs"] if kind==artifact["kind"])
                register_paths(conn, run_id, {artifact["kind"]: path}, copy=True, copied=copied)
    # In-core attachments are independently registered and must survive rebuild.
    paths = {kind: path for kind, path, _ in context["inputs"]}
    if "incore_manifest" in paths:
        from profile_db.ingest import incore, writer
        entries = incore.parse_manifest(paths["incore_manifest"].read_text())
        incore.merge_instr_metrics(entries, paths["instr_metrics"].read_text() if "instr_metrics" in paths else None)
        writer.insert_incore_entries(conn, run_id, entries)


def repair(db, *, run_id, source=None, dry_run=False):
    conn = db.connection
    cursor = conn.execute("SELECT * FROM run WHERE run_id=?", [run_id])
    names = [x[0] for x in cursor.description]
    row = cursor.fetchone()
    if not row:
        raise IngestError("repair run missing")
    meta = dict(zip(names, row))
    if conn.execute("SELECT 1 FROM diagnostic_protocol WHERE run_id=?", [run_id]).fetchone():
        from profile_db.ingest.diagnostic import repair as repair_diagnostic
        return repair_diagnostic(db, run_id=run_id, source=source, dry_run=dry_run)
    cursor = conn.execute("SELECT * FROM artifact WHERE run_id=? ORDER BY artifact_id", [run_id])
    columns = [x[0] for x in cursor.description]
    artifacts = cursor.fetchall()
    source_paths = {}
    if source is not None:
        from profile_db.ingest.source import discover_source
        from profile_db.ingest.analysis import discover_optional
        src = discover_source(source)
        source_paths = {src.records_kind: src.records, "deps": src.deps, "name_map": src.name_map,
                        **discover_optional(src.path)}
        if src.merged:
            source_paths["merged_swimlane"] = src.merged
        for kind, path in (("perf_hints", src.path.parent/"report/perf_hints.log"),
                           ("memory", src.path.parent/"report/memory_after_AllocateMemoryAddr.txt"),
                           ("pmu", src.path/"pmu.csv")):
            if path.is_file(): source_paths[kind] = path
    inputs, replacements = [], []
    for raw in artifacts:
        artifact = dict(zip(columns, raw))
        kind = artifact["kind"]
        try:
            path = resolve(conn, run_id, kind)
        except QueryError as exc:
            path = source_paths.get(kind)
            if path is None:
                raise IngestError("repair needs original capture/attachment: " + kind) from exc
        if digest(path) != artifact["sha256"]:
            raise IngestError("repair source does not match original: " + kind)
        inputs.append((kind, path, artifact["sha256"]))
        # Recovered linked paths are anchored to the supplied original bytes.
        artifact["mtime_ns"] = path.stat().st_mtime_ns
        if path == source_paths.get(kind) and artifact.get("store_mode") != "copy":
            import os
            from profile_db.runtime_adapter.storage import base_dir
            artifact["locator"] = os.path.relpath(path, base_dir(conn))
            artifact["mtime_ns"] = path.stat().st_mtime_ns
        replacements.append(tuple(artifact[k] for k in columns))
    paths = {k: v for k, v, _ in inputs}
    records = [k for k in RECORD_KINDS if k in paths]
    if len(records) != 1 or not {"deps", "name_map"} <= paths.keys():
        raise IngestError("repair requires anchored records, deps and name map")
    before = counts(conn, run_id)
    required = {"critical_path", "scheduler"}
    for family, document in conn.execute("SELECT family, document FROM analysis_snapshot WHERE run_id=?", [run_id]).fetchall():
        if json.loads(document).get("status") == "ok": required.add(family)
    context = dict(run_id=run_id, columns=columns, artifacts=replacements, original_artifacts=artifacts,
                   inputs=inputs, before=before, required_families=required)
    validate(context)
    if dry_run:
        return dict(run_id=run_id, status="ready", artifacts=len(inputs), before=before)
    identity_row = conn.execute("SELECT document FROM profile_identity WHERE run_id=?", [run_id]).fetchone()
    identity = json.loads(identity_row[0]) if identity_row else None
    requests = {m: dict(requested=r, value=v) for m,r,v in conn.execute(
        "SELECT modality, requested, request_value FROM modality_status WHERE run_id=?", [run_id]).fetchall()}
    with TemporaryDirectory(prefix="sluice-repair-") as directory:
        build = Path(directory)
        capture = build/"dfx_outputs"
        capture.mkdir()
        layout = {records[0]: capture/(records[0]+".json"), "deps": capture/"deps.json",
                  "name_map": capture/"name_map_repair.json", "pmu": capture/"pmu.csv",
                  "merged_swimlane": capture/"merged_swimlane_repair.json",
                  "perf_hints": build/"report/perf_hints.log",
                  "memory": build/"report/memory_after_AllocateMemoryAddr.txt",
                  "args_dump": capture/"args_dump/args_dump.json",
                  "scope_stats": capture/"scope_stats/scope_stats.jsonl"}
        for kind, target in layout.items():
            if kind in paths:
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind == 'name_map' and paths[kind].suffix == '.py':
                    from profile_db.runtime_adapter.kernel_names import document
                    target.write_text(json.dumps(document(paths[kind])))
                else:
                    shutil.copyfile(paths[kind], target)
        from profile_db.ingest import ingest_capture
        fields = ("program", "platform", "device_id", "git_commit", "git_dirty", "notes", "rank_label")
        kwargs = {k: meta[k] for k in fields}
        kwargs.update(captured_at=str(meta["captured_at"]) if meta["captured_at"] else None,
                      tags=list(meta["tags"] or []), runtime_cfg=json.loads(meta["runtime_cfg"] or "{}"))
        report = ingest_capture(db, capture, identity=identity, modality_requests=requests,
                                _repair_context=context, **kwargs)
    return dict(run_id=run_id, status="repaired", before=before, after=counts(conn, run_id), report=report)
