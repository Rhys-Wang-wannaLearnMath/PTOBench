# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Persisted analysis and read-only artifact resolution."""

from __future__ import annotations

import json
from pathlib import Path

from profile_db.errors import QueryError
from profile_db.runtime_adapter import ADAPTER_VERSION, ANALYSIS_VERSION, digest


def semantic_identity(doc):
    """Compare actual implementations independently of install locations."""
    keys = ("analysis_version", "adapter_version", "backend_revision", "functions",
            "compatibility", "parameters", "status")
    identity = {key: doc.get(key) for key in keys}
    identity["converter_sha256"] = (doc.get("backend") or {}).get("converter_sha256")
    identity["bundle_sha256"] = (doc.get("backend") or {}).get("bundle_sha256")
    identity["host_module_sha256"] = (doc.get("backend") or {}).get("host_module_sha256")
    return identity


def base_dir(conn):
    rows = conn.execute("SELECT path FROM duckdb_databases() WHERE NOT internal").fetchall()
    return Path(rows[0][0]).resolve().parent if rows and rows[0][0] else Path.cwd()


def read_snapshot(conn, run_id, family):
    row = conn.execute(
        "SELECT CAST(document AS VARCHAR) FROM analysis_snapshot WHERE run_id=? AND family=?",
        [run_id, family],
    ).fetchone()
    if not row:
        optional = {
            "args": "args_dump",
            "scope": "scope_stats",
            "incore": "incore_manifest",
            "host": "host_log:%",
        }
        if family in optional:
            present = conn.execute(
                "SELECT 1 FROM artifact WHERE run_id=? AND kind LIKE ? LIMIT 1", [run_id, optional[family]]
            ).fetchone()
            if not present:
                return dict(status="unavailable", reason="no registered artifact", family=family)
        return dict(status="stale", reason="refresh required", family=family)
    doc = json.loads(row[0])
    if doc["analysis_version"] != ANALYSIS_VERSION or doc.get("adapter_version") != ADAPTER_VERSION:
        return dict(status="stale", reason="analysis or adapter version changed; refresh required", family=family)
    return doc


def save_snapshot(conn, run_id, doc):
    conn.execute("DELETE FROM analysis_snapshot WHERE run_id=? AND family=?", [run_id, doc["family"]])
    conn.execute("INSERT INTO analysis_snapshot VALUES (?, ?, ?)", [run_id, doc["family"], json.dumps(doc)])


def resolve(conn, run_id, kind, *, bounded=False):
    rows = conn.execute(
        "SELECT locator, sha256, mtime_ns, size_bytes FROM artifact WHERE run_id=? AND kind=? ORDER BY artifact_id",
        [run_id, kind],
    ).fetchall()
    if len(rows) != 1 or not rows[0][0]:
        raise QueryError(f"artifact {kind}: absent or unanchored; attach/refresh required")
    path = base_dir(conn) / rows[0][0]
    if not path.is_file():
        raise QueryError(f"artifact {kind}: missing")
    if bounded:
        stat = path.stat()
        if rows[0][2] is None or stat.st_mtime_ns != rows[0][2] or stat.st_size != rows[0][3]:
            raise QueryError(f"artifact {kind}: changed; attach/refresh required")
    elif digest(path) != rows[0][1]:
        raise QueryError(f"artifact {kind}: changed; attach/refresh required")
    return path


def fingerprint(conn, run_id):
    artifacts = conn.execute(
        "SELECT kind, sha256 FROM artifact WHERE run_id=? ORDER BY kind, sha256", [run_id]
    ).fetchall()
    docs = conn.execute(
        "SELECT family, CAST(document AS VARCHAR) FROM analysis_snapshot WHERE run_id=? ORDER BY family",
        [run_id],
    ).fetchall()
    import hashlib

    return hashlib.sha256(json.dumps([artifacts, docs]).encode()).hexdigest()
