# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prune (DESIGN.md 8.1): delete every run outside the working set.

The row deletion runs inside one transaction under the writer lock, so a
failed prune leaves the database untouched. Copy-mode archive files and
render-cache directories for pruned runs are removed after the commit;
link mode copies nothing, so it deletes no files. ``trial`` and
``baseline`` rows are short-term memory and survive pruning.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from profile_db.db import WriterGuard
from profile_db.lifecycle.working_set import retained_run_ids

# Every run-scoped child table (DESIGN.md 5.2) except trial/baseline,
# which are memory and are deliberately kept.
_RUN_TABLES = (
    "clean_measurement",
    "diagnostic_metric",
    "diagnostic_protocol",
    "repair_audit",
    "profile_identity",
    "analysis_snapshot",
    "modality_status",
    "bench_stratum",
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
    "bench_sample",
    "incore_entry",
    "args_dump_entry",
    "scope_stats_entry",
)


def _all_run_ids(conn) -> set[int]:
    return {int(r[0]) for r in conn.execute("SELECT run_id FROM run").fetchall()}


def prune_runs(conn, db_path: Path | None, keep: int) -> dict[str, Any]:
    """Prune runs outside the retained set. Returns a report dict with
    ``kept``, ``pruned`` (sorted run-id lists), and ``removed_files``."""
    retained = retained_run_ids(conn, keep)
    pruned = sorted(_all_run_ids(conn) - retained)
    kept = sorted(retained & _all_run_ids(conn))

    with WriterGuard(db_path or ":memory:"):
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("UPDATE run SET retained = FALSE")
            if retained:
                placeholders = ",".join("?" * len(retained))
                conn.execute(
                    f"UPDATE run SET retained = TRUE WHERE run_id IN ({placeholders})",
                    list(retained),
                )
            for table in _RUN_TABLES:
                conn.execute(
                    f"DELETE FROM {table} WHERE run_id IN "
                    "(SELECT run_id FROM run WHERE retained = FALSE)"
                )
            conn.execute("DELETE FROM run WHERE retained = FALSE")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    removed_files: list[str] = []
    if db_path is not None:
        for run_id in pruned:
            for subdir in ("store", "render"):
                target = db_path.parent / subdir / str(run_id)
                if target.is_dir():
                    shutil.rmtree(target)
                    removed_files.append(str(target))
    return {"kept": kept, "pruned": pruned, "removed_files": removed_files}
