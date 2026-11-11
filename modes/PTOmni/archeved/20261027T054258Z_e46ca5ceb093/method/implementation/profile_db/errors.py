# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PFDB exception hierarchy."""


class PfdbError(Exception):
    """Base class for all profile_db errors."""


class DbError(PfdbError):
    """The database file cannot be opened or used."""


class MigrationError(PfdbError):
    """A schema migration is malformed or failed to apply."""


class LockError(PfdbError):
    """Another writer holds the database lock."""


class FactError(PfdbError):
    """A fact cannot be constructed or serialized."""


class IngestError(PfdbError):
    """A capture source is missing, malformed, or cannot be ingested."""


class QueryError(PfdbError):
    """A query cannot be answered: unknown name, invalid parameters, or a
    run/rank consistency check failed."""


class RenderError(PfdbError):
    """A render request cannot be satisfied: unknown kind, invalid window
    or parameters, or a target (run/task/core) that does not exist."""


class LifecycleError(PfdbError):
    """A lifecycle operation cannot proceed: incompatible comparison,
    unknown trial/baseline, or an invalid verdict/state transition."""
