# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Schema migration runner.

Migrations are ``NNNN_name.sql`` files under ``schema/migrations/``. The
runner keeps the applied-version bookkeeping in the ``schema_version``
table itself: each pending migration runs inside its own transaction and
is recorded only after it succeeds, so a failed migration leaves the
database at the previous version (rolled back, retryable).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from profile_db.errors import MigrationError

if TYPE_CHECKING:
    import duckdb

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_MIGRATION_NAME = re.compile(r"^(\d{4})_[A-Za-z0-9_]+\.sql$")

# Highest migration version shipped by this package.
SCHEMA_VERSION = 11
CAPTURE_PARSER_VERSION = "capture/2"
DIAGNOSTIC_PARSER_VERSION = "diagnostic/4"

# Artifact kinds that identify a run's records file (the accepted records
# filenames minus ``.json``). The records file's sha256 is the run identity
# used for idempotent ingest and for scoping the render cache; the tuple
# lives here so both the ingest and render layers can import it.
RECORD_KINDS = (
    "chip_swimlane_records",
    "l2_swimlane_records",
    "l2_perf_records",
)


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> dict[int, Path]:
    """Map ``version -> file`` for every migration in ``directory``."""
    found: dict[int, Path] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _MIGRATION_NAME.match(path.name)
        if match is None:
            raise MigrationError(
                f"invalid migration filename {path.name!r} (expected NNNN_name.sql)"
            )
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"duplicate migration version {version}")
        found[version] = path
    if not found:
        raise MigrationError(f"no migrations found under {directory}")
    return found


def current_version(conn: "duckdb.DuckDBPyConnection") -> int:
    """Highest applied migration version (0 when none applied yet)."""
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        ).fetchone()
    except Exception as exc:
        if "schema_version" in str(exc):
            return 0  # table not created yet
        raise
    return int(row[0])


def apply_pending(
    conn: "duckdb.DuckDBPyConnection", migrations_dir: Path = MIGRATIONS_DIR
) -> list[int]:
    """Apply pending migrations in order; returns the applied versions.

    Each migration is atomic: failure rolls the statement back and raises
    ``MigrationError`` without touching ``schema_version``. Applying twice
    is a no-op (idempotent open).
    """
    migrations = discover_migrations(migrations_dir)
    current = current_version(conn)
    # The order must be contiguous upward from the current version; the
    # minimum version of the set is the contract's start point.
    pending = sorted(v for v in migrations if v > current)
    applied: list[int] = []
    for version in pending:
        try:
            sql = migrations[version].read_text(encoding="utf-8")
        except OSError as exc:
            raise MigrationError(f"cannot read migration {version:04d}: {exc}") from exc
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, CURRENT_TIMESTAMP)",
                [version],
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            raise MigrationError(f"migration {version:04d} failed: {exc}") from exc
        applied.append(version)
    return applied


def available_versions(migrations_dir: Path = MIGRATIONS_DIR) -> Iterable[int]:
    """All versions shipped in ``migrations_dir``, ascending."""
    return sorted(discover_migrations(migrations_dir))
