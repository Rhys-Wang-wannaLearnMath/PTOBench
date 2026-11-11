# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Working-set determination (DESIGN.md 8.1).

The retained set is the latest ``keep`` runs plus every run referenced by
a baseline or by an active (``running``) trial. Runs outside the set are
prune candidates. This module reads only the ``run``/``baseline``/
``trial`` schema tables.
"""

from __future__ import annotations


def retained_run_ids(conn, keep: int) -> set[int]:
    """The run ids protected from pruning: latest ``keep`` (by run_id),
    every baseline-referenced run, and every active-trial-referenced run."""
    latest = {
        int(r[0])
        for r in conn.execute(
            "SELECT run_id FROM run ORDER BY run_id DESC LIMIT ?", [keep]
        ).fetchall()
    }
    baseline_refs = {
        int(r[0])
        for r in conn.execute(
            "SELECT DISTINCT run_id FROM baseline WHERE run_id IS NOT NULL"
        ).fetchall()
    }
    active_trial_refs = {
        int(r[0])
        for r in conn.execute(
            "SELECT DISTINCT run_id FROM trial WHERE run_id IS NOT NULL AND status = 'running'"
        ).fetchall()
    }
    import json
    group_refs = {m['run_id'] for (document,) in conn.execute(
        'SELECT CAST(document AS VARCHAR) FROM evidence_group').fetchall()
        for m in json.loads(document)['members']}
    return latest | baseline_refs | active_trial_refs | group_refs
