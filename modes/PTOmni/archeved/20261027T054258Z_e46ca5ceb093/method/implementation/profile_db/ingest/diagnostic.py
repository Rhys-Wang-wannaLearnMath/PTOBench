# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Own PMU/compiler ingestion independently of runtime swimlanes.

Every metric comes from an anchored artifact. Unknown or empty evidence is
unavailable; no runtime rows, clock calibration or benchmark latency is invented.
"""
from __future__ import annotations
import hashlib
import csv
import io
import json
import math
from pathlib import Path
from collections import Counter

from profile_db.errors import IngestError
from profile_db.identity import canonical_identity
from profile_db.ingest import text_evidence, writer
from profile_db.ingest.analysis import register_paths

from profile_db.schema import DIAGNOSTIC_PARSER_VERSION

PARSER_VERSION = DIAGNOSTIC_PARSER_VERSION


def discover(build, mode, allowed_root=None):
    build = Path(build)
    if not build.is_dir():
        raise IngestError("diagnostic build directory missing")
    root = Path(allowed_root or build).resolve()
    for path in (build, *build.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise IngestError("diagnostic artifact outside allowed root")
    patterns = {"pmu": "pmu.csv"} if mode == "pmu" else {
        "pipeline_profile": "pipeline_profile.json", "perf_hints": "perf_hints.log",
        "memory": "memory_after_AllocateMemoryAddr.txt",
        "allocation_ir": "*_after_AllocateMemoryAddr.py",
        "compiler_hardware": "compiler_hardware.json"}
    paths = {}
    for kind, pattern in patterns.items():
        found = sorted(build.rglob(pattern))
        if len(found) > 1:
            raise IngestError("ambiguous diagnostic artifact: " + kind)
        if found:
            paths[kind] = found[0]
    if mode == 'pmu':
        coverage = sorted(build.rglob('chip_swimlane_records.json'))
        if len(coverage) > 1:
            raise IngestError('ambiguous PMU coverage capture')
        if coverage:
            paths['pmu_coverage_records'] = coverage[0]
    if not paths:
        raise IngestError("diagnostic capture missing")
    return paths


def parse(paths, mode):
    metrics, parsed = [], {}
    if mode == "pmu":
        text = paths["pmu"].read_text()
        sample_count = len(list(csv.DictReader(io.StringIO(text))))
        rows = text_evidence.parse_pmu_csv(text)
        if sample_count == 0 or not rows:
            raise IngestError('empty PMU capture')
        parsed["pmu"] = rows
        totals = {}
        for r in rows:
            total = r["total_cycles"]
            if total is not None:
                previous = totals.setdefault(r["sample_seq"], total)
                if total != previous or not math.isfinite(total) or total < 0:
                    raise IngestError("invalid PMU total cycles")
        if totals and len(totals) == sample_count:
            metrics += [("pmu_total_cycles", sum(totals.values()), "cycles"),
                        ("pmu_samples", len(totals), "count")]
        signature = sorted(Counter((r["counter"], str(r["event_type"]), str(r["core_type"]),
                                    str(r["core_id"]), str(r["thread_id"])) for r in rows).items())
        if 'pmu_coverage_records' in paths:
            from profile_db.task_ids import normalize_task_id
            coverage = json.loads(paths['pmu_coverage_records'].read_text())
            from profile_db.runtime_adapter.capture import validate_capture
            validate_capture(coverage)  # Never pool independent executions.
            physical = coverage.get('aicore_tasks')
            metadata = coverage.get('metadata', {})
            threads = metadata.get('core_to_thread')
            if (not isinstance(physical,list) or not physical or not isinstance(threads,list)
                    or any(type(t) is not int or t < 0 for t in threads)):
                raise IngestError('invalid PMU coverage capture')
            expected = Counter()
            for row in physical:
                if not isinstance(row,list) or len(row) not in (5,6,7) or type(row[0]) is not int or not 0 <= row[0] < len(threads):
                    raise IngestError('invalid PMU coverage row')
                expected[(normalize_task_id(row[1]).canonical,row[0],threads[row[0]])] += 1
            observed = {r['sample_seq']:(r['task_id'],r['core_id'],r['thread_id']) for r in rows}
            if len(observed) != sample_count:
                raise IngestError('PMU sample identities missing')
            counters = {r['counter'] for r in rows}
            if len(rows) != sample_count*len(counters) or len(totals) != sample_count:
                raise IngestError('PMU coverage has missing counters or total cycles')
            actual = Counter(observed.values())
            missing = sum((expected-actual).values())
            extra = sum((actual-expected).values())
            complete = missing == extra == 0
            # A valid CSV can still miss dispatches or contain unrelated ones.
            # Keep that evidence for coverage diagnosis, never workload totals.
            if not complete:
                metrics = []
            metrics.extend([('pmu_expected_samples',sum(expected.values()),'count'),
                            ('pmu_observed_samples',sample_count,'count'),
                            ('pmu_missing_samples',missing,'count'),
                            ('pmu_extra_samples',extra,'count'),
                            ('pmu_coverage_complete',int(complete),'flag'),
                            ('pmu_coverage_fraction',sum((expected & actual).values())/sum(expected.values()),'fraction')])
            # Workload size and placement may change after fusion/splitting.
            # Verify completeness within each capture; compare acquisition
            # configuration independently of row multiplicity.
            signature = {'coverage':'physical_dispatch/2','counters':sorted(counters),
                'events':sorted({str(r['event_type']) for r in rows}),
                'clock_freq_hz':metadata.get('clock_freq_hz'), 'core_types':metadata.get('core_types')}
    else:
        signature = []
        if "pipeline_profile" in paths:
            doc = json.loads(paths["pipeline_profile"].read_text())
            seconds = doc.get("total_seconds")
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise IngestError("invalid pipeline total_seconds")
            stages = doc.get("stages")
            if not isinstance(stages, list) or not stages:
                raise IngestError("invalid pipeline stages")
            def validate_stages(items):
                for stage in items:
                    if not isinstance(stage, dict) or not isinstance(stage.get("name"), str) or not stage["name"]:
                        raise IngestError("invalid pipeline stage name")
                    duration = stage.get("seconds")
                    if type(duration) not in (int,float) or not math.isfinite(duration) or duration < 0:
                        raise IngestError("invalid pipeline stage duration")
                    children = stage.get("children", [])
                    if not isinstance(children, list): raise IngestError("invalid pipeline stage children")
                    validate_stages(children)
            validate_stages(stages)
            # total_seconds is full pipeline elapsed time, including execution
            # when the producer includes it. Never call it compilation latency.
            metrics.append(("pipeline_seconds", seconds, "s"))
            signature = sorted(s["name"] for s in stages)
        if "perf_hints" in paths:
            parsed["perf_hints"] = text_evidence.parse_perf_hints(paths["perf_hints"].read_text())
            metrics.append(("compiler_hint_count", len(parsed["perf_hints"]), "count"))
        if "memory" in paths:
            parsed["memory"] = text_evidence.parse_memory_report(paths["memory"].read_text())
            if parsed["memory"]:
                metrics.append(("buffer_peak_bytes", max(r["usage"] for r in parsed["memory"]), "bytes"))
        elif "allocation_ir" in paths:
            from profile_db.compiler_resources import read
            resources = read(paths["allocation_ir"], paths.get("compiler_hardware"))
            parsed["resources"] = resources
            parsed["memory"] = [{"kernel": k["kernel"], "space": s["space"],
                "usage": s["usage_bytes"], "limit_value": s["limit_bytes"]}
                for k in resources["kernels"] for s in k["spaces"] if s["complete"]]
            signature = [*signature, "resources:" + resources["parser_version"]]
            if resources["complete"]:
                # Legacy report entries were per-kernel/space footprints too.
                # Keep that metric distinct from the largest single allocation.
                metrics.append(("buffer_peak_bytes", max(r["usage"] for r in parsed["memory"]), "bytes"))
                largest = max(a["size_bytes"] for k in resources["kernels"] for a in k["allocations"])
                metrics.append(("largest_buffer_bytes", largest, "bytes"))
    return metrics, parsed, hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def ingest(db, source, *, mode, program, identity, allowed_root=None, copy=False):
    if mode not in ("pmu", "compiler"):
        raise IngestError("unsupported diagnostic mode")
    identity, key = canonical_identity(identity)
    if identity is None or identity.get("profile_mode") != mode:
        raise IngestError("diagnostic identity missing or mismatched")
    if copy and db.path is None:
        raise IngestError("copy archive mode requires a file-backed database")
    paths = discover(source, mode, allowed_root)
    try:
        metrics, parsed, signature = parse(paths, mode)
    except (ValueError, KeyError, TypeError) as exc:
        raise IngestError("diagnostic parse failed: " + str(exc)) from exc
    from profile_db.runtime_adapter import digest
    anchors = {k: digest(v) for k, v in paths.items()}
    copied = []
    try:
        with db._writing() as conn:
            row = conn.execute("SELECT run_id FROM run WHERE identity_key=?", [key]).fetchone()
            run_id = row[0] if row else writer.next_id(conn, "run", "run_id")
            if row:
                old = dict(conn.execute("SELECT kind, sha256 FROM artifact WHERE run_id=?", [run_id]).fetchall())
                if old != anchors:
                    raise IngestError("registered diagnostic action artifacts changed")
                protocol = conn.execute('SELECT parser_version FROM diagnostic_protocol WHERE run_id=?', [run_id]).fetchone()
                if protocol and protocol[0] == PARSER_VERSION:
                    return dict(run_id=run_id, mode=mode, metrics=len(metrics))
                # Reparse the exact original bytes under the new version in one
                # transaction. Preserve the immutable action and attachments.
                for table in ('diagnostic_metric', 'diagnostic_protocol', 'pmu_counter', 'perf_hint', 'memory_entry'):
                    conn.execute('DELETE FROM ' + table + ' WHERE run_id=?', [run_id])
            else:
                conn.execute("INSERT INTO run(run_id, program, identity_key, rank_label) VALUES (?, ?, ?, ?)",
                             [run_id, program, key, identity.get('rank_label','single')])
                conn.execute("INSERT INTO profile_identity VALUES (?, ?)", [run_id, json.dumps(identity, sort_keys=True)])
                register_paths(conn, run_id, paths, copy=copy, copied=copied)
            for metric, value, unit in metrics:
                conn.execute("INSERT INTO diagnostic_metric VALUES (?, ?, ?, ?)", [run_id, metric, value, unit])
            conn.execute("INSERT INTO diagnostic_protocol VALUES (?, ?, ?, ?)", [run_id, mode, PARSER_VERSION, signature])
            writer.insert_pmu_counters(conn, run_id, writer.next_id(conn, "pmu_counter", "pmu_id"), parsed.get("pmu", []))
            writer.insert_perf_hints(conn, run_id, parsed.get("perf_hints", []))
            writer.insert_memory_entries(conn, run_id, writer.next_id(conn, "memory_entry", "memory_id"), parsed.get("memory", []))
            # Rehash inside the transaction to refuse files changed during parsing.
            if any(digest(paths[k]) != v for k, v in anchors.items()):
                raise IngestError("diagnostic artifacts changed during ingest")
    except Exception:
        for path in copied: path.unlink(missing_ok=True)
        raise

    return dict(run_id=run_id, mode=mode, metrics=len(metrics))


def repair(db, *, run_id, source=None, dry_run=False):
    """Rebuild diagnostic rows from exact registered originals in one transaction."""
    from profile_db.errors import QueryError
    from profile_db.runtime_adapter import digest
    from profile_db.runtime_adapter.storage import resolve
    conn = db.connection
    protocol = conn.execute('SELECT mode FROM diagnostic_protocol WHERE run_id=?', [run_id]).fetchone()
    if not protocol:
        raise IngestError('diagnostic protocol missing')
    mode = protocol[0]
    if mode not in ('pmu', 'compiler'):
        raise IngestError('diagnostic repair supports PMU/compiler; clean requires its original benchmark receipt')
    artifacts = conn.execute('SELECT kind,sha256,store_mode FROM artifact WHERE run_id=? ORDER BY kind', [run_id]).fetchall()
    replacements = discover(source, mode) if source is not None else {}
    paths, recovered = {}, []
    for kind, sha, storage in artifacts:
        try:
            path = resolve(conn, run_id, kind)
        except QueryError as exc:
            path = replacements.get(kind)
            if path is None:
                raise IngestError('diagnostic repair needs original artifact: ' + kind) from exc
            recovered.append((kind, path, storage))
        if digest(path) != sha:
            raise IngestError('diagnostic repair source does not match original: ' + kind)
        paths[kind] = path
    anchors = {kind: sha for kind, sha, _ in artifacts}
    def validate():
        if any(not p.is_file() or digest(p) != anchors[k] for k, p in paths.items()):
            raise IngestError('diagnostic repair input changed or missing')
    validate()
    try:
        metrics, parsed, signature = parse(paths, mode)
    except (ValueError, KeyError, TypeError) as exc:
        raise IngestError('diagnostic repair parse failed: ' + str(exc)) from exc
    tables = ('diagnostic_metric', 'diagnostic_protocol', 'pmu_counter', 'perf_hint', 'memory_entry')
    def counts():
        return {t: conn.execute('SELECT count(*) FROM ' + t + ' WHERE run_id=?', [run_id]).fetchone()[0]
                for t in tables}
    before = counts()
    validate()
    if dry_run:
        return {'run_id':run_id, 'status':'would_repair', 'mode':mode, 'before':before, 'metrics':len(metrics)}
    copied = []
    try:
        with db._writing() as conn:
            validate()
            for table in tables:
                conn.execute('DELETE FROM ' + table + ' WHERE run_id=?', [run_id])
            for metric, value, unit in metrics:
                conn.execute('INSERT INTO diagnostic_metric VALUES (?, ?, ?, ?)', [run_id, metric, value, unit])
            conn.execute('INSERT INTO diagnostic_protocol VALUES (?, ?, ?, ?)', [run_id, mode, PARSER_VERSION, signature])
            writer.insert_pmu_counters(conn, run_id, writer.next_id(conn, 'pmu_counter', 'pmu_id'), parsed.get('pmu', []))
            writer.insert_perf_hints(conn, run_id, parsed.get('perf_hints', []))
            writer.insert_memory_entries(conn, run_id, writer.next_id(conn, 'memory_entry', 'memory_id'), parsed.get('memory', []))
            for kind, path, storage in recovered:
                register_paths(conn, run_id, {kind:path}, copy=storage == 'copy', copied=copied)
            validate()
    except Exception:
        for path in copied:
            path.unlink(missing_ok=True)
        raise
    return {'run_id':run_id, 'status':'repaired', 'mode':mode, 'before':before, 'after':counts()}
