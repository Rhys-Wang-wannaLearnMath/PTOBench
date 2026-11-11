# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The layered query engine (DESIGN.md 6, T4).

Importing this package registers every query with the registry. The public
surface is ``execute`` / ``list_queries`` / ``get_query``: run a named
query, get back a budget-limited facts stream. Handlers read only schema
tables and the derived layer's tested pure functions — never the ingest
parsers or the raw JSON artifacts — so no raw swimlane JSON can leak into
an answer.
"""

from __future__ import annotations

from profile_db.query import (  # noqa: F401  (import side-effect: registration)
    handlers_z0,
    handlers_z1,
    handlers_z2,
    handlers_z3,
    handlers_z4,
    handlers_modalities,
    handlers_performance,
)
from profile_db.query.registry import (
    QuerySpec,
    execute,
    get_query,
    list_queries,
    register,
)
from profile_db.query.result import QueryOutput

__all__ = [
    "execute",
    "get_query",
    "list_queries",
    "register",
    "QueryOutput",
    "QuerySpec",
]