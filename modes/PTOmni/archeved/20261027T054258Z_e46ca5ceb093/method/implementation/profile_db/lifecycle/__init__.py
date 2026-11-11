# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The lifecycle & short-term-memory layer (DESIGN.md 8, T8).

Working-set determination and prune read/write only schema tables plus
the filesystem (copy-store / render-cache removal); trial/baseline/compare
are the short-term memory that survives pruning. This package never
touches the ingest parsers or the raw JSON artifacts.
"""

from __future__ import annotations

from profile_db.lifecycle.baseline import add_baseline, diff_baseline, list_baselines
from profile_db.lifecycle.compare import compare_runs
from profile_db.lifecycle.ids import next_id
from profile_db.lifecycle.prune import prune_runs
from profile_db.lifecycle.trial import (
    attach_bench,
    bind_trial,
    list_trials,
    register_trial,
    set_verdict,
)
from profile_db.lifecycle.working_set import retained_run_ids


__all__ = [
    "add_baseline",
    "attach_bench",
    "bind_trial",
    "compare_runs",
    "diff_baseline",
    "list_baselines",
    "list_trials",
    "prune_runs",
    "register_trial",
    "retained_run_ids",
    "set_verdict",
    "next_id",
]
