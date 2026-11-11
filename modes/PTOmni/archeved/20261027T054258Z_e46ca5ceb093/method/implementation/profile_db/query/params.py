# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Query parameter models — the single source the CLI/MCP entries are
generated from (DESIGN.md 6.6/9).

Every registered query owns exactly one pydantic model here, so tool
schemas, CLI arguments, and validation stay one definition. Models are
deliberately thin: semantic guards that need database context (run
existence, run/rank consistency, and window ordering) live in the handlers
and raise ``QueryError``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from profile_db.task_ids import normalize_task_id


class PfdbParams(BaseModel):
    """Shared base: forbid unknown keys so typos surface loudly."""

    model_config = ConfigDict(extra="forbid")


class RunsListParams(PfdbParams):
    rank: str | None = Field(default=None, description="rank_label filter")


class RunIdParams(PfdbParams):
    run_id: int
    rank: str | None = Field(
        default=None,
        description="require this run to belong to the selected rank label",
    )


class OverviewParams(RunIdParams):
    pass


class ProfileDetailParams(RunIdParams):
    view: str = Field(pattern='^(tasks|critical_path|engine_occupancy|dispatch_delays|scheduler_phases|pmu|resources|allocations|hints|samples|pipeline|host)$')
    offset: int = Field(default=0, ge=0, le=10000000)
    limit: int = Field(default=8, ge=1, le=16)


class InventoryParams(RunIdParams):
    analysis_details: bool = Field(default=False, description="include full analysis provenance and function fingerprints")


class DensityParams(RunIdParams):
    engine: str | None = Field(default=None, description="restrict to one engine")
    bands: int = Field(default=20, ge=1, le=1000, description="display buckets")


class SparseRegionsParams(RunIdParams):
    engine: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)


class WhySparseParams(RunIdParams):
    # Two ways to address the window, exactly one required (the handler
    # raises QueryError otherwise). Both coordinates are exact integers —
    # never round-trip a display-rounded µs bound, which would drag in the
    # neighbouring band.
    #  - band + bands: the display bucket axis of ``density``. ``bands``
    #    MUST equal the value passed to ``density``, or the bucket index
    #    denotes a different window.
    #  - stored_band: the 5 µs storage band index, which is what
    #    ``sparse_regions`` reports as ``stored_band_idx``. It is NOT
    #    interchangeable with ``band``.
    band: int | None = Field(
        default=None, ge=0, description="display bucket index from density"
    )
    bands: int = Field(
        default=20, ge=1, le=1000, description="display bucket count density used"
    )
    stored_band: int | None = Field(
        default=None, ge=0, description="storage band index from sparse_regions"
    )
    engine: str | None = None


class RegionParams(RunIdParams):
    t0_us: float
    t1_us: float
    family: str | None = None
    core: int | None = None


class CoreParams(RunIdParams):
    core: int = Field(ge=0)


class IdleWindowParams(RunIdParams):
    after_task_id: str = Field(
        description="producer whose end bounds the window start"
    )
    until_task_id: str | None = Field(
        default=None,
        description="consumer whose start bounds the window end "
        "(default: next observed-CP successor)",
    )
    engine: str | None = Field(
        default=None,
        description="engine to measure (default: opposite of the producer)",
    )

    @field_validator("after_task_id")
    @classmethod
    def _canonical_after_task_id(cls, value: str) -> str:
        return normalize_task_id(value).canonical

    @field_validator("until_task_id")
    @classmethod
    def _canonical_until_task_id(cls, value: str | None) -> str | None:
        return normalize_task_id(value).canonical if value is not None else None


class TaskIdParams(RunIdParams):
    task_id: str

    @field_validator("task_id")
    @classmethod
    def _canonical_task_id(cls, value: str) -> str:
        return normalize_task_id(value).canonical


class TaskParams(TaskIdParams):
    pass


class TasksParams(RunIdParams):
    family: str | None = Field(
        default=None, description="exact family name (at least one of family/name required)"
    )
    name: str | None = Field(
        default=None, description="substring match on task name"
    )
    engine: str | None = Field(default=None, description="restrict to one engine")
    on_cpm: str | None = Field(
        default=None,
        pattern="^(observed|static)$",
        description="restrict to observed or static critical-path members",
    )


class DepsParams(RunIdParams):
    task_id: str | None = None
    view: str = Field(default="direct", pattern="^(direct|reduced|dataflow)$")

    @field_validator("task_id")
    @classmethod
    def _optional_task(cls, value):
        return normalize_task_id(value).canonical if value is not None else None

    direction: str = Field(default="out", pattern="^(in|out|all)$")


class SubgraphParams(TaskIdParams):
    depth: int = Field(default=2, ge=1, le=6)
    max_nodes: int = Field(default=24, ge=1, le=200)


class WhyLateParams(TaskIdParams):
    pass


class WhyLongParams(TaskIdParams):
    pass


class RowsParams(TaskIdParams):
    pass


class SchedulerParams(RunIdParams):
    task_id: str | None = None
    view: str = Field(default="summary", pattern="^(summary|phases|cores|rows)$")

    @field_validator("task_id")
    @classmethod
    def _optional_task(cls, value):
        return normalize_task_id(value).canonical if value is not None else None



class EarlyDispatchParams(TaskIdParams):
    pass


class PmuParams(TaskIdParams):
    samples: bool = Field(
        default=False,
        description="include one provenance-preserving fact for each PMU CSV sample",
    )


class CriticalPathParams(RunIdParams):
    kind: str = Field(default="observed", pattern="^(observed|static)$")


class PerfHintsParams(RunIdParams):
    pass


class MemoryParams(RunIdParams):
    pass


class IncoreParams(RunIdParams):
    kernel: str | None = Field(default=None, description="restrict to one kernel")
    view: str = Field(default="summary", pattern="^(summary|events)$")
    pipe: str | None = None
    t0_us: float | None = None
    t1_us: float | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=64, ge=1, le=256)


class ArgsDumpParams(RunIdParams):
    view: str = Field(default="metadata", pattern="^(metadata|values)$")
    seq: int | None = Field(default=None, ge=0)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=64, ge=1, le=256)
    task_id: str | None = Field(default=None, description="restrict to one task")
    stage: str | None = Field(default=None, description="restrict to one stage")

    @field_validator("task_id")
    @classmethod
    def _canonical_optional_task_id(cls, value: str | None) -> str | None:
        return normalize_task_id(value).canonical if value is not None else None


class HostParams(RunIdParams):
    view: str = Field(default="summary", pattern="^(summary|rounds|tree|cpu)$")
    pid: int | None = None
    inv: int | None = None
    hid: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=64, ge=1, le=256)


class ScopeStatsParams(RunIdParams):
    view: str = Field(default="summary", pattern="^(summary|records)$")
    ring: int | None = None
    site: str | None = Field(default=None, description="restrict to one site")


class BenchParams(RunIdParams):
    pass
