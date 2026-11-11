# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Argument metadata rows; payload registration/decoding use the shared artifact context."""

from __future__ import annotations

import json
from typing import Any

from profile_db.task_ids import normalize_task_id


def parse_args_dump(text: str) -> list[dict[str, Any]]:
    """Parse ``args_dump.json`` text into metadata rows (seq-ordered)."""
    doc = json.loads(text)
    if not isinstance(doc, dict):
        return []
    args = doc.get("args", doc.get("tensors", []))
    if not isinstance(args, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, arg in enumerate(args):
        if not isinstance(arg, dict):
            continue
        shape = arg.get("shape")
        identity = normalize_task_id(arg["task_id"]) if arg.get("task_id") is not None else None
        rows.append(
            {
                "seq": index,
                "task_id": identity.canonical if identity is not None else None,
                "task_id_raw": identity.raw if identity is not None else None,
                "task_id_u64": identity.u64 if identity is not None else None,
                "stage": arg.get("stage"),
                "role": arg.get("role"),
                "arg_index": arg.get("arg_index"),
                "kind": arg.get("kind", "tensor"),
                "dtype": arg.get("dtype"),
                "shape": shape if isinstance(shape, list) else [],
                "bin_size": int(arg.get("bin_size") or 0),
            }
        )
    return rows
