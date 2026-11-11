# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Database connection management and the writer lock.

Lifecycle (DESIGN.md 5.1):

- one embedded DuckDB file per repository, default
  ``<cwd>/.pfdb/profile.duckdb`` or ``$PFDB_PATH`` when set;
- ``ProfileDB.memory()`` runs the exact same schema on ``:memory:``
  (single-shot working set, zero disk footprint, disposal by dropping
  the handle);
- every write path opens inside an exclusive, fail-fast ``WriterGuard``
  (``flock`` on ``<db>.lock``). ``flock`` locks the open file
  description, so two connections inside one process conflict just like
  two processes: a second writer gets ``LockError`` immediately instead
  of queueing.

The database is a disposable working set: corrupt files are reported
with a rebuild hint rather than an attempt to repair (DESIGN.md 1.3).
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any

import duckdb

from profile_db.errors import DbError, LockError
from profile_db.schema import apply_pending, current_version

DEFAULT_DB_DIR = ".pfdb"
DEFAULT_DB_NAME = "profile.duckdb"


def default_db_path() -> Path:
    """Resolution order: ``$PFDB_PATH``, else ``<cwd>/.pfdb/profile.duckdb``."""
    env = os.environ.get("PFDB_PATH")
    if env:
        return Path(env)
    return Path.cwd() / DEFAULT_DB_DIR / DEFAULT_DB_NAME


def _is_memory(path: Path | str | None) -> bool:
    return path is not None and str(path) == ":memory:"


class WriterGuard:
    """Exclusive, fail-fast write lock: ``flock(LOCK_EX | LOCK_NB)`` on
    ``<db>.lock``. No-op for "":memory:"" databases."""

    def __init__(self, db_path: Path | str | None):
        self._memory = _is_memory(db_path)
        self._lock_path = None if self._memory else Path(f"{db_path}.lock")
        self._fd: int | None = None

    def __enter__(self) -> "WriterGuard":
        if self._lock_path is None:
            return self
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise LockError(
                f"database is locked by another writer: {self._lock_path}"
            ) from exc
        self._fd = fd
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        return False


class ProfileDB:
    """One PFDB database handle: opens (creating when needed) and applies
    pending schema migrations under the writer lock. Exposes the raw
    DuckDB connection for milestone modules (ingest/derived/query)."""

    def __init__(self, path: Path | str | None = None, *, read_only: bool = False):
        if _is_memory(path):
            self._path: Path | None = None
            self._conn = duckdb.connect(":memory:", read_only=False)
            apply_pending(self._conn)
            return
        db_path = Path(path) if path is not None else default_db_path()
        db_path = db_path.expanduser()
        self._path = db_path
        if read_only:
            try:
                self._conn = duckdb.connect(str(self._path), read_only=True)
            except (duckdb.IOException, duckdb.ConnectionException) as exc:
                _raise_open_error(self._path, exc)
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with WriterGuard(self._path):
            try:
                self._conn = duckdb.connect(str(self._path), read_only=False)
            except (duckdb.IOException, duckdb.ConnectionException) as exc:
                _raise_open_error(self._path, exc)
            else:
                apply_pending(self._conn)

    @classmethod
    def memory(cls) -> "ProfileDB":
        """In-memory database with the same schema (single-shot working set)."""
        return cls(":memory:")

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    @property
    def path(self) -> Path | None:
        """Database file path (``None`` for :memory:)."""
        return self._path

    def schema_version(self) -> int:
        return current_version(self._conn)

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "ProfileDB":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False


def _raise_open_error(path: Path, exc: Exception) -> None:
    """Translate DuckDB's process-level file lock into the PFDB lock error."""
    detail = str(exc)
    if (
        "conflicting lock" in detail.lower()
        or "could not set lock" in detail.lower()
        or "same database file with a different configuration" in detail.lower()
    ):
        raise LockError(
            f"database is locked by another DuckDB process: {path}; wait for its "
            "write connection to close and retry"
        ) from exc
    raise DbError(
        f"cannot open pfdb at {path}: {detail}. "
        "This database is a disposable working set; rebuild it from source artifacts."
    ) from exc
