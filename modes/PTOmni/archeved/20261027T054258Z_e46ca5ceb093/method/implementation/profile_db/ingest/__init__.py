# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Capture ingest: the public entry that turns one ``dfx_outputs/``
directory into run/task/task_row/edge/phase rows.

Design contracts implemented here (DESIGN.md 5.1/5.3, T1 acceptance):

- default ``store_mode=link``: artifacts are registered with sha256 and
  size but never copied; ``copy=True`` archives them inside
  ``.pfdb/store/<run_id>/``;
- idempotent re-ingest: the run is identified by the records-file
  sha256; re-ingesting replaces that run's rows inside one transaction
  (delete children, update run, re-insert), so row counts are stable;
- the whole write is atomic: any failure rolls back to the pre-ingest
  state (and removes archive copies created by this attempt);
- the AICore<->AICPU join happens inside ``ingest.swimlane``, which
  delegates to the upstream converter.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from profile_db.db import WriterGuard
from profile_db.derived import derive_run
from profile_db.errors import IngestError
from profile_db.ingest import (
    args_dump,
    deps,
    scope_stats,
    source as source_mod,
    swimlane,
    text_evidence,
    writer,
)

_CHILD_TABLES = (
    "profile_identity",
    "analysis_snapshot",
    "artifact",
    "task",
    "task_row",
    "dep_edge",
    "scheduler_phase",
    "orch_phase",
    "perf_hint",
    "memory_entry",
    "pmu_counter",
    "time_band",
    "idle_gap",
    "cpm_path",
    "args_dump_entry",
    "scope_stats_entry",
    "modality_status",
    "bench_stratum",
    # Re-ingesting a capture rebuilds the whole run, and it deletes the
    # artifact rows that registered the in-core manifest — so the in-core
    # rows must go with them, or incore_entry would keep claiming an
    # attachment that `inventory` can no longer show. Re-attach with
    # `pfdb ingest-incore` after a re-ingest.
    "incore_entry",
)

# Artifact kinds owned by `ingest_incore` rather than by the capture
# ingest, so its replace-mode delete can be scoped to them.
_INCORE_ARTIFACT_KINDS = ("incore_manifest", "instr_metrics")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_captured_at(value: str) -> str:
    """Compact ``YYYYMMDD_HHMMSS`` directory timestamps -> SQL timestamp
    text; free-form values pass through for DuckDB to cast."""
    try:
        return datetime.strptime(value, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return value


def _artifact(kind: str, path: Path, rel: str, store_mode: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path,
        "rel_path": rel,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "store_mode": store_mode,
    }


def _merge_task_timing(graph: deps.DepGraph, swimlane_run: swimlane.Swimlane) -> None:
    """Attach swimlane aggregates to every logical task; rows whose tokens
    are missing from deps.json are a structural surprise and fail loudly."""
    groups: dict[str, list[swimlane.RowTime]] = {}
    for row in swimlane_run.rows:
        groups.setdefault(row.task_id, []).append(row)
    unknown = sorted(set(groups) - graph.task_ids)
    if unknown:
        raise IngestError(f"swimlane contains rows for tasks absent from deps.json: {unknown[:5]}")
    for task in graph.tasks:
        rows = groups.get(task.task_id)
        if not rows:
            continue
        engines = {row.engine for row in rows}
        task.engine = next(iter(engines)) if len(engines) == 1 else "mixed"
        task.num_rows = len(rows)
        min_start = min(row.start_us for row in rows)
        max_end = max(row.end_us for row in rows)
        min_dispatch = min(row.dispatch_us for row in rows)
        min_receive = min(row.receive_us for row in rows)
        max_finish = max(row.finish_us for row in rows)
        task.busy_us = max_end - min_start
        task.wall_us = max_finish - min_dispatch
        task.min_dispatch_us = min_dispatch
        task.min_receive_us = min_receive
        task.min_start_us = min_start
        task.max_end_us = max_end
        task.max_finish_us = max_finish


def ingest_capture(
    db,
    source: Path | str,
    *,
    program: str | None = None,
    platform: str | None = None,
    device_id: int | None = None,
    captured_at: str | None = None,
    notes: str | None = None,
    tags: Sequence[str] | None = None,
    rank_label: str | None = None,
    copy: bool = False,
    runtime_cfg: dict[str, Any] | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    bench_min_us: float | None = None,
    bench_median_us: float | None = None,
    bench_mean_us: float | None = None,
    bench_max_us: float | None = None,
    bench_rounds: int | None = None,
    bench_strata: Sequence[dict[str, Any]] | None = None,
    modality_requests: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
    allowed_root: Path | None = None,
    _repair_context=None,
) -> dict[str, Any]:
    """Ingest one capture directory into ``db``; returns a summary dict.

    Optional text-modality evidence is auto-discovered: ``report/
    perf_hints.log`` and ``report/memory_after_AllocateMemoryAddr.txt``
    beside the capture, plus ``dfx_outputs/pmu.csv`` inside it. Missing
    evidence files simply leave their tables empty for the run (the
    query layer reports those as unavailable). Benchmark numbers are
    registered through the ``bench_*`` parameters.

    Raises ``IngestError`` for missing/malformed artifacts or an unusable
    environment; the database is left exactly as before the failed call.
    """
    started = time.perf_counter()
    cpu_started = time.process_time()
    from profile_db.identity import canonical_identity
    identity, identity_key = canonical_identity(identity)
    db_bytes_before = _db_size_bytes(db.path)
    src = source_mod.discover_source(source)
    if allowed_root is not None:
        allowed_root = Path(allowed_root).resolve()
        for path in (src.records,src.deps,src.name_map):
            if not path.resolve().is_relative_to(allowed_root) or any(p.is_symlink() for p in (path,*path.parents)):
                raise IngestError('artifact outside allowed root')
        for path in src.path.parent.rglob('*'):
            if path.is_symlink() or not path.resolve().is_relative_to(allowed_root):
                raise IngestError('artifact outside allowed root')
        for path in discover_bounded_paths(src.path):
            if not path.resolve().is_relative_to(allowed_root):
                raise IngestError('attachment outside allowed root')
    records = source_mod.raw_records(src.records)
    level = source_mod.records_level(records)  # validates 1..4
    name_map = source_mod.load_name_map(src.name_map)
    swimlane_run = swimlane.load(src.records, records)
    graph = deps.load_deps(src.deps, name_map)
    _merge_task_timing(graph, swimlane_run)

    if program is None:
        program = src.program
    if program is None:
        raise IngestError(f"cannot determine program name; pass --program (source: {src.path})")
    if captured_at is None:
        captured_at = src.captured_at
    if captured_at is not None:
        captured_at = _normalize_captured_at(captured_at)

    store_mode = "copy" if copy else "link"
    if copy and db.path is None:
        raise IngestError("copy archive mode requires a file-backed database")

    artifacts: list[dict[str, Any]] = [
        _artifact(src.records_kind, src.records, source_mod.rel_path(src, src.records), store_mode),
        _artifact("deps", src.deps, source_mod.rel_path(src, src.deps), store_mode),
        _artifact("name_map", src.name_map, source_mod.rel_path(src, src.name_map), store_mode),
    ]
    if src.merged is not None:
        artifacts.append(
            _artifact("merged_swimlane", src.merged, source_mod.rel_path(src, src.merged), store_mode)
        )

    # Rank isolation (DESIGN.md 5.3): a multi-rank capture set must label
    # its runs. Once any labelled run exists, a new unlabelled ingest would
    # silently join the 'single' bucket and mix ranks, so it is refused.
    # Re-ingesting a capture already in the database is exempt — it inherits
    # its own label.
    if rank_label is None and writer.find_run_by_records(db.connection, artifacts[0]["sha256"], identity_key) is None:
        labelled = db.connection.execute(
            "SELECT DISTINCT rank_label FROM run WHERE rank_label <> 'single' ORDER BY rank_label"
        ).fetchall()
        if labelled:
            raise IngestError(
                "this database already holds rank-labelled runs "
                f"({', '.join(str(r[0]) for r in labelled)}); pass rank_label/--rank "
                "so ranks stay isolated"
            )

    # Optional text-modality evidence and extended modalities: absence and a
    # parser failure are capture facts, not reasons to discard the core trace.
    report_dir = src.path.parent / "report"
    perf_hints_path = report_dir / "perf_hints.log" if report_dir.is_dir() else None
    memory_path = report_dir / "memory_after_AllocateMemoryAddr.txt" if report_dir.is_dir() else None
    pmu_path = src.path / "pmu.csv"
    args_dump_path = src.path / "args_dump" / "args_dump.json"
    scope_stats_path = src.path / "scope_stats" / "scope_stats.jsonl"

    def _read_optional(path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestError(f"{path}: cannot read: {exc}") from exc

    manifest_requests = _capture_manifest_requests(src.path)
    requests = {**manifest_requests, **(modality_requests or {})}
    optional = {
        "perf_hints": (perf_hints_path, text_evidence.parse_perf_hints),
        "memory": (memory_path, text_evidence.parse_memory_report),
        "pmu": (pmu_path, text_evidence.parse_pmu_csv),
        "args_dump": (args_dump_path, args_dump.parse_args_dump),
        "scope_stats": (scope_stats_path, scope_stats.parse_scope_stats),
    }
    parsed_optional: dict[str, list[dict[str, Any]]] = {}
    modality_statuses: list[dict[str, Any]] = []
    for modality, (path, parser) in optional.items():
        requested = _request_bool(requests.get(modality))
        request_value = _request_value(requests.get(modality))
        if path is None or not path.is_file():
            parsed_optional[modality] = []
            modality_statuses.append(
                _modality_status(modality, requested, request_value, None, None, "absent", None)
            )
            continue
        try:
            rows = parser(_read_optional(path) or "")
        except (IngestError, ValueError, json.JSONDecodeError) as exc:
            parsed_optional[modality] = []
            modality_statuses.append(
                _modality_status(
                    modality,
                    requested,
                    request_value,
                    source_mod.rel_path(src, path),
                    path.stat().st_size,
                    "parse_error",
                    None,
                    str(exc),
                )
            )
            continue
        parsed_optional[modality] = rows
        modality_statuses.append(
            _modality_status(
                modality,
                requested,
                request_value,
                source_mod.rel_path(src, path),
                path.stat().st_size,
                "parsed",
                len(rows),
            )
        )
    for modality in ("l2_swimlane", "dep_gen"):
        requested = _request_bool(requests.get(modality))
        modality_statuses.append(
            _modality_status(
                modality, requested, _request_value(requests.get(modality)), None, None, "core", 1
            )
        )
    perf_hints_rows = parsed_optional["perf_hints"]
    memory_rows = parsed_optional["memory"]
    pmu_rows = parsed_optional["pmu"]
    args_dump_rows = parsed_optional["args_dump"]
    scope_stats_rows = parsed_optional["scope_stats"]
    for kind, path in (
        ("perf_hints", perf_hints_path),
        ("memory", memory_path),
        ("pmu", pmu_path),
        ("args_dump", args_dump_path),
        ("scope_stats", scope_stats_path),
    ):
        if isinstance(path, Path) and path.is_file():
            artifacts.append(_artifact(kind, path, source_mod.rel_path(src, path), store_mode))

    # runtime_cfg carries the artifact-declared orchestrator source: a
    # host-orchestrated capture splices a second clock domain, and µs
    # values live on the causal composite the converter built.
    runtime_cfg = dict(runtime_cfg or {})
    runtime_cfg["orchestrator_source"] = swimlane_run.orchestrator_source

    meta = {
        "program": program,
        "platform": platform,
        "device_id": device_id,
        "captured_at": captured_at,
        "swimlane_level": level,
        "clock_freq_hz": swimlane_run.clock_freq_hz,
        "num_cores": swimlane_run.num_cores,
        "core_types": swimlane_run.core_types,
        "core_to_thread": swimlane_run.core_to_thread,
        "rank_label": rank_label,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runtime_cfg": runtime_cfg,
        "bench_min_us": bench_min_us,
        "bench_median_us": bench_median_us,
        "bench_mean_us": bench_mean_us,
        "bench_max_us": bench_max_us,
        "bench_rounds": bench_rounds,
        "makespan_us": swimlane_run.makespan_us,
        "raw_span_us": swimlane_run.raw_span_us,
        "notes": notes,
        "tags": list(tags or []),
    }

    copied: list[Path] = []

    def _archive(target_dir: Path, name: str, artifact_path: Path) -> str:
        dest = target_dir / (_sha256(artifact_path) + "-" + name)
        if not dest.exists():
            shutil.copy2(artifact_path, dest)
            copied.append(dest)
        elif _sha256(dest) != _sha256(artifact_path):
            raise IngestError("archive contents changed")
        return f"store/{target_dir.name}/{dest.name}"

    conn = db.connection
    with WriterGuard(db.path or ":memory:"):
        conn.execute("BEGIN TRANSACTION")
        try:
            if _repair_context is not None:
                current = conn.execute("SELECT * FROM artifact WHERE run_id=? ORDER BY artifact_id", [_repair_context["run_id"]]).fetchall()
                if current != _repair_context["original_artifacts"]:
                    raise IngestError("repair registration changed; retry with a fresh preview")
            records_sha = artifacts[0]["sha256"]
            run_id = writer.find_run_by_records(conn, records_sha, identity_key)
            if _repair_context is not None and run_id != _repair_context["run_id"]:
                raise IngestError("repair run identity changed")
            if run_id is not None:
                _carry_forward(conn, run_id, meta)
                child_tables = _CHILD_TABLES
                if bench_strata is None:
                    # Raw benchmarks are independent of the profiled capture.
                    # Preserve a previously ingested authoritative benchmark
                    # when the user merely re-ingests its DFX artifacts.
                    child_tables = tuple(table for table in _CHILD_TABLES if table != "bench_stratum")
                writer.delete_run_rows(conn, run_id, child_tables)
                writer.update_run(conn, run_id, meta)
            else:
                run_id = writer.next_id(conn, "run", "run_id")
                meta = {**meta, "run_id": run_id}
                writer.insert_run(conn, meta)

            if copy:
                target_dir = db.path.parent / "store" / str(run_id)
                target_dir.mkdir(parents=True, exist_ok=True)
                for artifact in artifacts:
                    artifact["rel_path"] = _archive(target_dir, Path(artifact["path"]).name, artifact["path"])

            writer.insert_artifacts(
                conn,
                run_id,
                writer.next_id(conn, "artifact", "artifact_id"),
                [a for a in artifacts],
            )
            from profile_db.ingest.analysis import collect, discover_optional, persist, register_paths
            from profile_db.runtime_adapter.storage import base_dir

            for artifact in artifacts:
                location = base_dir(conn) / artifact["rel_path"] if copy else artifact["path"]
                conn.execute(
                    "UPDATE artifact SET locator=?, mtime_ns=? WHERE run_id=? AND kind=?",
                    [
                        os.path.relpath(location, base_dir(conn)),
                        location.stat().st_mtime_ns,
                        run_id,
                        artifact["kind"],
                    ],
                )
            register_paths(conn, run_id, discover_optional(src.path), copy=copy, copied=copied)
            if _repair_context is not None:
                from profile_db.ingest.repair import restore_artifacts
                restore_artifacts(conn, run_id, _repair_context, copied)
            docs = collect(conn, run_id)
            if _repair_context is not None and any(d["status"] != "ok" for d in docs if d["family"] in _repair_context["required_families"]):
                raise IngestError("repair analysis failed; original run preserved")
            persist(conn, run_id, docs)
            writer.insert_tasks(conn, run_id, _task_rows(graph))
            writer.insert_task_rows(conn, run_id, _task_row_rows(swimlane_run))
            writer.insert_edges(
                conn,
                run_id,
                writer.next_id(conn, "dep_edge", "edge_id"),
                graph.edges,
            )
            writer.insert_scheduler_phases(
                conn,
                run_id,
                writer.next_id(conn, "scheduler_phase", "phase_id"),
                swimlane_run.scheduler_phases,
            )
            writer.insert_orchestrator_phases(conn, run_id, swimlane_run.orchestrator_phases)
            writer.insert_perf_hints(conn, run_id, perf_hints_rows)
            writer.insert_memory_entries(
                conn,
                run_id,
                writer.next_id(conn, "memory_entry", "memory_id"),
                memory_rows,
            )
            writer.insert_pmu_counters(
                conn,
                run_id,
                writer.next_id(conn, "pmu_counter", "pmu_id"),
                pmu_rows,
            )
            writer.insert_args_dump_entries(conn, run_id, args_dump_rows)
            writer.insert_scope_stats_entries(conn, run_id, scope_stats_rows)
            writer.insert_modality_statuses(conn, run_id, modality_statuses)
            from profile_db.ingest.analysis import sync_modality_statuses

            sync_modality_statuses(conn, run_id)
            if bench_strata is not None:
                conn.execute("DELETE FROM bench_sample WHERE run_id = ?", [run_id])
                conn.execute("DELETE FROM bench_stratum WHERE run_id = ?", [run_id])
                writer.insert_bench_strata(conn, run_id, bench_strata)
                writer.insert_bench_samples(
                    conn,
                    run_id,
                    [
                        {"stratum": item["stratum"], "round": round_index, "effective_us": value}
                        for item in bench_strata
                        for round_index, value in enumerate(item["samples"])
                    ],
                )
            derivation = derive_run(conn, run_id)
            writer.insert_time_bands(conn, run_id, derivation.bands)
            writer.insert_idle_gaps(
                conn,
                run_id,
                writer.next_id(conn, "idle_gap", "gap_id"),
                derivation.gaps,
            )
            writer.insert_cpm_paths(conn, run_id, derivation.paths)
            writer.update_task_path_flags(conn, run_id, derivation.task_flags)
            writer.update_run_cpm(conn, run_id, derivation.cpm_us)
            from profile_db.schema import CAPTURE_PARSER_VERSION
            conn.execute("UPDATE run SET identity_key=?, capture_parser_version=? WHERE run_id=?", [identity_key, CAPTURE_PARSER_VERSION, run_id])
            if identity is not None:
                conn.execute("INSERT INTO profile_identity VALUES (?, ?)", [run_id, json.dumps(identity, sort_keys=True, allow_nan=False)])
            if _repair_context is not None:
                from profile_db.ingest.repair import validate, counts
                validate(_repair_context)
                audit = dict(before=_repair_context["before"], after=counts(conn, run_id),
                             anchors={k: sha for k, _, sha in _repair_context["inputs"]})
                conn.execute("INSERT INTO repair_audit(run_id, document) VALUES (?, ?)", [run_id, json.dumps(audit, sort_keys=True)])
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            for path in copied:
                path.unlink(missing_ok=True)
            if isinstance(exc, IngestError):
                raise
            raise IngestError(f"ingest transaction failed: {exc}") from exc

    db_bytes_after = _db_size_bytes(db.path)
    peak_rss, rss_source = _peak_rss_bytes()
    return {
        "run_id": run_id,
        "program": program,
        "level": level,
        "tasks": len(graph.tasks),
        "task_rows": len(swimlane_run.rows),
        "edges": len(graph.edges),
        "artifacts": int(
            conn.execute("SELECT count(*) FROM artifact WHERE run_id=?", [run_id]).fetchone()[0]
        ),
        "perf_hints": len(perf_hints_rows),
        "memory_entries": len(memory_rows),
        "pmu_counters": len(pmu_rows),
        "args_dump": len(args_dump_rows),
        "scope_stats": len(scope_stats_rows),
        "store_mode": store_mode,
        "makespan_us": swimlane_run.makespan_us,
        "time_bands": len(derivation.bands),
        "idle_gaps": len(derivation.gaps),
        "cpm_path": len(derivation.paths),
        "cpm_us": derivation.cpm_us,
        "bench_strata": len(bench_strata or ()),
        "bench_samples": sum(len(item["samples"]) for item in (bench_strata or ())),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "cpu_ms": round((time.process_time() - cpu_started) * 1000.0, 3),
        "peak_rss_bytes": peak_rss,
        "rss_source": rss_source,
        "db_bytes_before": db_bytes_before,
        "db_bytes_after": db_bytes_after,
        "db_growth_bytes": db_bytes_after - db_bytes_before,
    }


def _db_size_bytes(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def discover_bounded_paths(source):
    from profile_db.ingest.analysis import discover_optional
    return discover_optional(source).values()


def _peak_rss_bytes() -> tuple[int, str]:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "posix":
        return int(value) * 1024, "getrusage.ru_maxrss_kib"
    return int(value), "getrusage.ru_maxrss_bytes"


def _capture_manifest_requests(capture_dir: Path) -> dict[str, Any]:
    paths = (
        capture_dir / "profile_capture_manifest.json",
        capture_dir.parent / "profile_capture_manifest.json",
    )
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    modalities = payload.get("modalities") if isinstance(payload, dict) else None
    if not isinstance(modalities, dict):
        return {}
    values: dict[str, Any] = {}
    for name, item in modalities.items():
        if isinstance(item, dict) and "requested" in item:
            values[str(name)] = item
    return values


def _request_bool(value: Any) -> bool | None:
    if isinstance(value, dict):
        return bool(value.get("requested")) if "requested" in value else None
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")


def _request_value(value: Any) -> str | None:
    if isinstance(value, dict):
        item = value.get("value")
    else:
        item = value
    if item is None:
        return None
    return str(item)


def _modality_status(
    modality: str,
    requested: bool | None,
    request_value: str | None,
    rel_path: str | None,
    size_bytes: int | None,
    parser_state: str,
    entry_count: int | None,
    reason: str | None = None,
) -> dict[str, Any]:
    if parser_state == "parse_error":
        state = "parse_error"
    elif parser_state == "core":
        state = "available"
    elif rel_path is None and requested is True:
        state = "not_emitted"
    elif rel_path is None and requested is False:
        state = "not_requested"
    elif rel_path is None:
        state = "unknown_request"
    elif entry_count == 0:
        state = "empty"
    else:
        state = "available"
    if reason is None:
        reason = {
            "not_emitted": "requested but artifact was not emitted",
            "not_requested": "not requested and artifact is absent",
            "unknown_request": "request state is unknown and artifact is absent",
            "empty": "artifact parsed successfully but contains no entries",
        }.get(state)
    return {
        "modality": modality,
        "requested": requested,
        "request_value": request_value,
        "rel_path": rel_path,
        "size_bytes": size_bytes,
        "parser_state": parser_state,
        "entry_count": entry_count,
        "state": state,
        "reason": reason,
    }


def _carry_forward(conn, run_id: int, meta: dict[str, Any]) -> None:
    """On re-ingest, merge the run columns that do not come from the
    artifacts with what the run already has: explicitly supplied values
    win, missing ones keep the previous registration.

    Benchmark numbers belong to the unprofiled measurement, which is
    independent of the artifacts being replaced; ``rank_label`` identifies
    the capture's rank, which a re-ingest must not silently reset to
    'single'.
    """
    carried = (
        "bench_min_us",
        "bench_median_us",
        "bench_mean_us",
        "bench_max_us",
        "bench_rounds",
        "rank_label",
    )
    row = conn.execute(
        "SELECT bench_min_us, bench_median_us, bench_mean_us, bench_max_us, "
        "bench_rounds, rank_label FROM run WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if row is None:
        return
    existing = dict(zip(carried, row))
    meta.update({key: (meta.get(key) if meta.get(key) is not None else existing.get(key)) for key in carried})


def _task_rows(graph: deps.DepGraph) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "task_id_raw": task.task_id_raw,
            "name": task.name,
            "family": task.family,
            "engine": task.engine,
            "scope": task.scope,
            "early_dispatch": task.early_dispatch,
            "kernel_ids": task.kernel_ids,
            "block_num": task.block_num,
            "num_rows": task.num_rows,
            "busy_us": task.busy_us,
            "wall_us": task.wall_us,
            "min_dispatch_us": task.min_dispatch_us,
            "min_receive_us": task.min_receive_us,
            "min_start_us": task.min_start_us,
            "max_end_us": task.max_end_us,
            "max_finish_us": task.max_finish_us,
        }
        for task in graph.tasks
    ]


def _task_row_rows(swimlane_run: swimlane.Swimlane) -> list[dict[str, Any]]:
    return [
        {
            "task_id": row.task_id,
            "task_id_raw": row.task_id_raw,
            "core_id": row.core_id,
            "engine": row.engine,
            "thread": row.thread,
            "row_index": row.row_index,
            "start_us": row.start_us,
            "end_us": row.end_us,
            "dispatch_us": row.dispatch_us,
            "receive_us": row.receive_us,
            "finish_us": row.finish_us,
        }
        for row in swimlane_run.rows
    ]


def ingest_incore(db, source, *, run_id):
    """Compatibility alias for the generic existing-artifact attachment."""
    from profile_db.ingest.analysis import attach

    return attach(db, source, run_id=run_id)
