# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""pfdb: the agent-oriented profiling & feedback database for PyPTO.

Query-first, data-disposable (DESIGN.md): the database is a working set
built from ``build_output`` artifacts and can be pruned or rebuilt at
any time.

Layered imports are enforced by ``.importlinter`` at the repo root and
by the AST layering test in ``tests/``; keep the dependency direction:
``cli/mcp -> db -> schema`` and ``facts -> errors``.
"""

from ._version import __version__

__all__ = ["__version__"]
