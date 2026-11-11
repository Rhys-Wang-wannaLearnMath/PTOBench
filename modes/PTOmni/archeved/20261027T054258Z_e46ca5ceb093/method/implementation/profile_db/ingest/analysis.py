# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Analyze only registered, existing artifacts; never collect profiling data."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from profile_db.errors import IngestError
from profile_db.ingest import args_dump, incore, scope_stats, writer
from profile_db.runtime_adapter import critical_path, dependencies, digest, scheduler, scope, snapshot
from profile_db.runtime_adapter.modalities import args, host, incore as incore_snapshot
from profile_db.runtime_adapter.storage import base_dir, read_snapshot, resolve, save_snapshot
from profile_db.schema import RECORD_KINDS


def discover_optional(source):
    source = Path(source)
    candidates = [source] if source.is_file() else sorted(source.rglob("*"))
    paths = {}
    for path in candidates:
        if not path.is_file():
            continue
        name = path.name
        kind = {
            "args_dump.json": "args_dump",
            "scope_stats.jsonl": "scope_stats",
            "manifest_export.csv": "incore_manifest",
            "instr_metrics.json": "instr_metrics",
            "dispatch_identity.json": "dispatch_identity",
            "dispatch_pairing.json": "dispatch_pairing",
        }.get(name)
        if name.startswith("host.") and name.endswith(".log"):
            kind = "host_log:" + name
        elif name == "host_phase_records.jsonl":
            kind = "host_phase:" + str(path.relative_to(source) if source.is_dir() else path.name)
        if kind:
            if kind in paths:
                raise IngestError(f"ambiguous attachment: {kind}")
            paths[kind] = path.resolve()
    if "args_dump" in paths:
        try:
            doc = json.loads(paths["args_dump"].read_text())
            payload = (
                paths["args_dump"].parent / doc.get("bin_file", "args.bin") if isinstance(doc, dict) else None
            )
            if payload is not None and payload.is_file():
                paths["args_payload"] = payload.resolve()
        except (ValueError, TypeError):
            pass
    if "incore_manifest" in paths:
        import csv

        with paths["incore_manifest"].open() as stream:
            for row in csv.DictReader(stream):
                kernel = row.get("kernel", row.get("kernel_name", row.get("func", "")))
                for field in ("trace_json", "visualize_data"):
                    value = row.get(field)
                    if value:
                        trace = Path(value)
                        if not trace.is_absolute():
                            trace = paths["incore_manifest"].parent / trace
                        if trace.is_file() and trace.suffix == ".json":
                            paths["incore_trace:" + kernel] = trace.resolve()
    return paths


def register_paths(conn, run_id, paths, *, copy=False, copied=None):
    copied = [] if copied is None else copied
    for kind, path in paths.items():
        sha = digest(path)
        locator = os.path.relpath(path, base_dir(conn))
        if copy:
            target = base_dir(conn) / "store" / str(run_id) / (sha + "-" + path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != sha:
                raise IngestError(f"archive artifact changed: {target.name}")
            if not target.exists():
                shutil.copy2(path, target)
                copied.append(target)
            locator = os.path.relpath(target, base_dir(conn))
        existing = conn.execute(
            "SELECT artifact_id FROM artifact WHERE run_id=? AND kind=?", [run_id, kind]
        ).fetchall()
        if existing:
            conn.execute("DELETE FROM artifact WHERE run_id=? AND kind=?", [run_id, kind])
        artifact_id = writer.next_id(conn, "artifact", "artifact_id")
        writer.insert_artifacts(
            conn,
            run_id,
            artifact_id,
            [
                dict(
                    kind=kind,
                    rel_path=path.name,
                    sha256=sha,
                    size_bytes=path.stat().st_size,
                    store_mode="copy" if copy else "link",
                )
            ],
        )
        registered = base_dir(conn) / locator
        conn.execute(
            "UPDATE artifact SET locator=?, mtime_ns=? WHERE artifact_id=?",
            [locator, registered.stat().st_mtime_ns, artifact_id],
        )
    return copied


def collect(conn, run_id, families=None):
    def relevant(kind):
        if families is None:
            return True
        return (
            (kind.startswith(("host_log:", "host_phase:")) and "host" in families)
            or (kind in ("args_dump", "args_payload") and "args" in families)
            or (kind == "scope_stats" and "scope" in families)
            or ((kind.startswith("incore") or kind == "instr_metrics") and "incore" in families)
        )

    paths = {}
    for (kind,) in conn.execute(
        "SELECT DISTINCT kind FROM artifact WHERE run_id=? AND locator IS NOT NULL", [run_id]
    ).fetchall():
        if relevant(kind):
            paths[kind] = resolve(conn, run_id, kind)
    records = next((paths[k] for k in RECORD_KINDS if k in paths), None)
    docs = []

    def add(family, inputs, fn, parameters=None):
        docs.append(snapshot(family, inputs, fn, parameters))

    if records is not None and "deps" in paths:
        inputs = dict(records=records, deps=paths["deps"])
        if "name_map" in paths:
            add(
                "critical_path",
                dict(inputs, name_map=paths["name_map"]),
                lambda c: critical_path(records, paths["deps"], paths["name_map"], c),
                {"tol_ticks": 2},
            )
        add("scheduler", inputs, lambda c: scheduler(records, paths["deps"], c))
    if "deps" in paths:
        add("deps", {"deps": paths["deps"]}, lambda c: dependencies(paths["deps"], c))
    if "scope_stats" in paths:
        add("scope", {"scope_stats": paths["scope_stats"]}, lambda c: scope(paths["scope_stats"], c))
    if "args_dump" in paths:
        add(
            "args",
            {k: paths[k] for k in ("args_dump", "args_payload") if k in paths},
            lambda c: args(paths["args_dump"]),
        )
    if "incore_manifest" in paths:
        add(
            "incore",
            {k: v for k, v in paths.items() if k.startswith("incore") or k == "instr_metrics"},
            lambda c: incore_snapshot(paths["incore_manifest"], c),
        )
    host_paths = {k: v for k, v in paths.items() if k.startswith(("host_log:", "host_phase:"))}
    if host_paths:
        add("host", host_paths, lambda c: host(host_paths, c))
    return docs


def persist(conn, run_id, docs):
    # Retire derived simulator state when an old run is refreshed. Originals
    # remain archived; no runtime-window simulation is rerun.
    conn.execute("DELETE FROM analysis_snapshot WHERE run_id=? AND family='wait'", [run_id])
    for doc in docs:
        save_snapshot(conn, run_id, doc)


def sync_modality_statuses(conn, run_id):
    for family, kind, payload_key in (
        ("args", "args_dump", "entries"),
        ("scope", "scope_stats", "records"),
        ("incore", "incore_manifest", "kernels"),
    ):
        artifact = conn.execute(
            "SELECT rel_path,size_bytes FROM artifact WHERE run_id=? AND kind=?", [run_id, kind]
        ).fetchone()
        if artifact is None:
            continue
        doc = read_snapshot(conn, run_id, family)
        old = conn.execute(
            "SELECT requested,request_value FROM modality_status WHERE run_id=? AND modality=?",
            [run_id, kind],
        ).fetchone()
        count = len(doc["payload"][payload_key]) if doc["status"] == "ok" else None
        if family == "scope" and count is not None:
            count += 1
        conn.execute("DELETE FROM modality_status WHERE run_id=? AND modality=?", [run_id, kind])
        writer.insert_modality_statuses(
            conn,
            run_id,
            [
                dict(
                    modality=kind,
                    requested=old[0] if old else None,
                    request_value=old[1] if old else None,
                    rel_path=artifact[0],
                    size_bytes=artifact[1],
                    entry_count=count,
                    parser_state="parsed" if count is not None else "parse_error",
                    state=("available" if count else "empty") if count is not None else "parse_error",
                    reason=doc.get("reason")
                    if count != 0
                    else "artifact parsed successfully but contains no entries",
                )
            ],
        )


def attach(db, source, *, run_id, copy=False):
    if db.connection.execute("SELECT 1 FROM run WHERE run_id=?", [run_id]).fetchone() is None:
        raise IngestError(f"run {run_id} does not exist")
    if copy and db.path is None:
        raise IngestError("copy archive mode requires a file-backed database")
    paths = discover_optional(source)
    if not paths:
        raise IngestError("source has no supported existing artifacts")
    copied = []
    try:
        with db._writing() as conn:
            families = set()
            if "args_dump" in paths:
                families.add("args")
                conn.execute(
                    "DELETE FROM artifact WHERE run_id=? AND kind IN ('args_dump','args_payload')", [run_id]
                )
                conn.execute("DELETE FROM args_dump_entry WHERE run_id=?", [run_id])
            if "scope_stats" in paths:
                families.add("scope")
                conn.execute("DELETE FROM scope_stats_entry WHERE run_id=?", [run_id])
            if "incore_manifest" in paths:
                families.add("incore")
                conn.execute(
                    "DELETE FROM artifact WHERE run_id=? AND (kind LIKE 'incore%' OR kind='instr_metrics')",
                    [run_id],
                )
                conn.execute("DELETE FROM incore_entry WHERE run_id=?", [run_id])
            if any(k.startswith(("host_log:", "host_phase:")) for k in paths):
                families.add("host")
            register_paths(conn, run_id, paths, copy=copy, copied=copied)
            docs = collect(conn, run_id, families)
            statuses = {d["family"]: d["status"] for d in docs}
            if "args_dump" in paths and statuses.get("args") == "ok":
                conn.execute("DELETE FROM args_dump_entry WHERE run_id=?", [run_id])
                writer.insert_args_dump_entries(
                    conn, run_id, args_dump.parse_args_dump(paths["args_dump"].read_text())
                )
            if "scope_stats" in paths and statuses.get("scope") == "ok":
                conn.execute("DELETE FROM scope_stats_entry WHERE run_id=?", [run_id])
                writer.insert_scope_stats_entries(
                    conn, run_id, scope_stats.parse_scope_stats(paths["scope_stats"].read_text())
                )
            entries = []
            if "incore_manifest" in paths and statuses.get("incore") == "ok":
                entries = incore.parse_manifest(paths["incore_manifest"].read_text())
                incore.merge_instr_metrics(
                    entries, paths["instr_metrics"].read_text() if "instr_metrics" in paths else None
                )
                conn.execute("DELETE FROM incore_entry WHERE run_id=?", [run_id])
                writer.insert_incore_entries(conn, run_id, entries)
            persist(conn, run_id, docs)
            sync_modality_statuses(conn, run_id)
    except Exception:
        for path in copied:
            path.unlink(missing_ok=True)
        raise
    return dict(
        run_id=run_id,
        artifacts=len(paths),
        incore_entries=len(entries),
        exported=sum(e.get("status") == "exported" for e in entries),
        analyses={d["family"]: d["status"] for d in docs},
    )


def refresh(db, *, run_id, source=None):
    from profile_db.derived import derive_run
    from profile_db.ingest.source import discover_source

    with db._writing() as conn:
        if conn.execute("SELECT 1 FROM run WHERE run_id=?", [run_id]).fetchone() is None:
            raise IngestError(f"run {run_id} does not exist")
        from profile_db.schema import CAPTURE_PARSER_VERSION
        parsed = conn.execute("SELECT capture_parser_version FROM run WHERE run_id=?", [run_id]).fetchone()[0]
        if parsed != CAPTURE_PARSER_VERSION:
            raise IngestError("physical capture rows require repair, not refresh; use repair --run-id N [--source ORIGINAL]")
        if source is not None:
            src = discover_source(source)
            old = conn.execute(
                "SELECT sha256 FROM artifact WHERE run_id=? AND kind IN "
                "('chip_swimlane_records','l2_perf_records','l2_swimlane_records')",
                [run_id],
            ).fetchone()
            if not old or digest(src.records) != old[0]:
                raise IngestError("refresh source must match the original capture identity")
            for kind, path in (("deps", src.deps), ("name_map", src.name_map)):
                anchor = conn.execute(
                    "SELECT sha256 FROM artifact WHERE run_id=? AND kind=?", [run_id, kind]
                ).fetchone()
                if anchor and digest(path) != anchor[0]:
                    raise IngestError(f"refresh {kind} must match the original artifact")
            register_paths(
                conn, run_id, {src.records_kind: src.records, "deps": src.deps, "name_map": src.name_map}
            )
            optional = discover_optional(src.path)
            for kind, path in optional.items():
                anchor = conn.execute(
                    "SELECT sha256 FROM artifact WHERE run_id=? AND kind=?", [run_id, kind]
                ).fetchone()
                if anchor and digest(path) != anchor[0]:
                    raise IngestError(
                        f"refresh {kind} must match the original artifact; use attach for replacement"
                    )
            register_paths(conn, run_id, optional)
        docs = collect(conn, run_id)
        if not any(d["family"] == "critical_path" for d in docs):
            raise IngestError("refresh requires anchored core artifacts; use --source")
        for doc in docs:
            previous = conn.execute(
                "SELECT CAST(document AS VARCHAR) FROM analysis_snapshot WHERE run_id=? AND family=?",
                [run_id, doc["family"]],
            ).fetchone()
            if doc["status"] != "ok" and previous and json.loads(previous[0])["status"] == "ok":
                raise IngestError(f"refresh failed for {doc['family']}: {doc.get('reason')}")
        persist(conn, run_id, docs)
        sync_modality_statuses(conn, run_id)
        result = derive_run(conn, run_id)
        for table in ("time_band", "idle_gap", "cpm_path"):
            conn.execute(f"DELETE FROM {table} WHERE run_id=?", [run_id])
        writer.insert_time_bands(conn, run_id, result.bands)
        writer.insert_idle_gaps(conn, run_id, writer.next_id(conn, "idle_gap", "gap_id"), result.gaps)
        writer.insert_cpm_paths(conn, run_id, result.paths)
        writer.update_task_path_flags(conn, run_id, result.task_flags)
        writer.update_run_cpm(conn, run_id, result.cpm_us)
    return dict(run_id=run_id, analyses={d["family"]: d["status"] for d in docs})
