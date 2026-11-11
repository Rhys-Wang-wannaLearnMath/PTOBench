# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Surrogate-id helper for the lifecycle layer (trial/baseline memory rows
are kept forever, so ids are assigned as ``max(id) + 1`` like the ingest
writers do). Kept in its own leaf module so the lifecycle submodules can
import it without a package-level circular import."""

from __future__ import annotations


def next_id(conn, table: str, column: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX({column}), 0) FROM {table}").fetchone()
    return int(row[0]) + 1
