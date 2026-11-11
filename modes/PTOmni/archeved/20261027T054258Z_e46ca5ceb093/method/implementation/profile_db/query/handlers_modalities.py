# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Persisted modality summaries and bounded registered-file slices."""

from __future__ import annotations

from typing import Any

from profile_db.facts import Evidence, Fact
from profile_db.query import common, runtime
from profile_db.query.params import (
    ArgsDumpParams,
    BenchParams,
    IncoreParams,
    ScopeStatsParams,
    HostParams,
    RunIdParams,
)
from profile_db.query.registry import register


@register(
    "incore",
    "Constraints: what metrics did the in-core simulator report for this "
    "kernel, and did the export succeed?",
    IncoreParams,
)
def incore(conn, params: IncoreParams) -> list[Fact]:
    if params.view == "events":
        return runtime.incore_events(conn, params)
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("INCORE", run_id)
    sql = (
        "SELECT kernel, status, export_dir, CAST(metrics AS VARCHAR) "
        "FROM incore_entry WHERE run_id = ?"
    )
    args: list[Any] = [run_id]
    if params.kernel is not None:
        sql += " AND kernel = ?"
        args.append(params.kernel)
    rows = common.q(conn, sql + " ORDER BY kernel", args)
    if not rows:
        facts = [
            Fact(
                "INCORE",
                common.fields(run_id=run_id, kernel=params.kernel),
                Evidence.UNAVAILABLE,
            )
        ]
        return facts
    return [
        Fact(
            "INCORE",
            common.fields(
                run_id=run_id,
                kernel=kernel,
                status=status,
                export_dir=export_dir,
                metrics=common.json_cell(metrics),
            ),
            Evidence.MEASURED,
        )
        for kernel, status, export_dir, metrics in rows
    ]


@register(
    "args_dump",
    "Constraints: which tensors were captured at the kernel boundary, with "
    "what dtype/shape?",
    ArgsDumpParams,
)
def args_dump(conn, params: ArgsDumpParams) -> list[Fact]:
    status = common.modality_status_fact(conn, params.run_id, "args_dump")
    return [*([status] if status is not None else []), *runtime.args_dump(conn, params)]


@register(
    "scope_stats",
    "Attribute: at which site/ring/phase do the scope-level stats sit, and "
    "what is the payload?",
    ScopeStatsParams,
)
def scope_stats(conn, params: ScopeStatsParams) -> list[Fact]:
    status = common.modality_status_fact(conn, params.run_id, "scope_stats")
    return [*([status] if status is not None else []), *runtime.scope_stats(conn, params)]


@register(
    "bench",
    "Verify: what is the unprofiled PYPTO_BENCH number (kept strictly apart "
    "from makespan)?",
    BenchParams,
)
def bench(conn, params: BenchParams) -> list[Fact]:
    """The unprofiled benchmark summary. DESIGN.md 5.3 keeps this strictly
    apart from ``makespan_us``: bench has no observer overhead, makespan
    does, and the two must never be compared to each other."""
    run_id = params.run_id
    row = common.one(
        conn,
        "SELECT bench_min_us, bench_median_us, bench_mean_us, bench_max_us, "
        "bench_rounds FROM run WHERE run_id = ?",
        [run_id],
    )
    if row is None:
        return common.run_missing("BENCH", run_id)
    if all(value is None for value in row):
        return [
            Fact(
                "BENCH",
                common.fields(
                    run_id=run_id,
                    reason="no bench numbers registered; ingest with --bench/--bench-log",
                ),
                Evidence.UNAVAILABLE,
            )
        ]
    facts = [
        Fact(
            "BENCH",
            common.fields(
                run_id=run_id,
                min_us=common.us(row[0]),
                median_us=common.us(row[1]),
                mean_us=common.us(row[2]),
                max_us=common.us(row[3]),
                rounds=row[4],
            ),
            Evidence.MEASURED,
        )
    ]
    samples = common.q(
        conn,
        "SELECT stratum, round, effective_us FROM bench_sample "
        "WHERE run_id = ? ORDER BY stratum, round",
        [run_id],
    )
    facts.extend(
        Fact(
            "BENCH_SAMPLE",
            common.fields(
                run_id=run_id, stratum=stratum, round=rnd, effective_us=common.us(value)
            ),
            Evidence.MEASURED,
        )
        for stratum, rnd, value in samples
    )
    return facts


@register("host", "Attribute: which host phases, invocations and CPU/off-CPU costs explain this capture?", HostParams)
def host(conn, params: HostParams) -> list[Fact]:
    return runtime.host(conn, params)


@register("diagnostic_summary", "Measured PMU/compiler metrics without runtime assumptions.", RunIdParams)
def diagnostic_summary(conn, params):
    if common.one(conn, "SELECT 1 FROM run WHERE run_id=?", [params.run_id]) is None:
        return common.run_missing("METRIC", params.run_id)
    rows = common.q(conn, "SELECT metric, value, unit FROM diagnostic_metric WHERE run_id=? ORDER BY metric", [params.run_id])
    ir = common.one(conn, "SELECT 1 FROM artifact WHERE run_id=? AND kind='allocation_ir'", [params.run_id])
    return [Fact("METRIC", {metric: value, "unit": unit},
                 Evidence.PROVEN if ir and metric in ('buffer_peak_bytes', 'largest_buffer_bytes') else Evidence.MEASURED)
            for metric, value, unit in rows] or [Fact("METRIC", {"reason": "diagnostic metric unavailable"}, Evidence.UNAVAILABLE)]


@register("compiler_resources", "Allocated address extent, single buffers and hardware headroom per kernel/space.", RunIdParams)
def compiler_resources(conn, params):
    from profile_db.compiler_resources import read
    from profile_db.runtime_adapter.storage import resolve
    ir = common.one(conn, "SELECT 1 FROM artifact WHERE run_id=? AND kind='allocation_ir'", [params.run_id])
    if not ir:
        return [Fact("RESOURCE", {"reason": "allocation IR unavailable"}, Evidence.UNAVAILABLE)]
    present = common.one(conn, "SELECT 1 FROM artifact WHERE run_id=? AND kind='compiler_hardware'", [params.run_id])
    report = read(resolve(conn, params.run_id, 'allocation_ir'),
                  resolve(conn, params.run_id, 'compiler_hardware') if present else None)
    facts = []
    for kernel in report['kernels']:
        if not kernel['spaces']:
            facts.append(Fact('RESOURCE', {'kernel': kernel['kernel'], 'errors': kernel['errors'],
                                          'reason': 'allocation spaces unavailable'}, Evidence.UNAVAILABLE))
        if not kernel['allocations']:
            facts.append(Fact('ALLOCATION', {'kernel': kernel['kernel'],
                                            'reason': 'allocations unavailable'}, Evidence.UNAVAILABLE))
        for space in kernel['spaces']:
            facts.append(Fact('RESOURCE', {**space, 'kernel': kernel['kernel'],
                'kernel_signature': kernel['signature'], 'errors': kernel['errors']},
                Evidence.PROVEN if space['complete'] else Evidence.UNAVAILABLE))
        for allocation in kernel['allocations']:
            facts.append(Fact('ALLOCATION', {**allocation, 'kernel': kernel['kernel']},
                Evidence.PROVEN if kernel['allocations_complete'] and allocation['size_bytes'] is not None else Evidence.UNAVAILABLE))
    return facts or [Fact('RESOURCE', {'reason': 'no local allocations'}, Evidence.UNAVAILABLE)]


@register("measurement_summary", "Clean latency distribution with validated public provenance.", RunIdParams)
def measurement_summary(conn, params):
    row = common.one(conn, 'SELECT CAST(document AS VARCHAR) FROM clean_measurement WHERE run_id=?', [params.run_id])
    if not row:
        return [Fact('METRIC', {'reason': 'clean measurement unavailable'}, Evidence.UNAVAILABLE)]
    import json
    summary = json.loads(row[0])['summary']
    return [Fact('METRIC', {key: value}, Evidence.MEASURED)
            for key, value in summary.items() if key != 'samples_us']


@register('compiler_resource_summary', 'Worst compiler memory-space headroom and single allocation, keeping their meanings separate.', RunIdParams)
def compiler_resource_summary(conn, params):
    resources = compiler_resources(conn, params)
    spaces = [f for f in resources if f.rec == 'RESOURCE']
    allocations = [f for f in resources if f.rec == 'ALLOCATION']
    values = {}
    if allocations and all(f.evidence == Evidence.PROVEN for f in allocations):
        values['largest_buffer_bytes'] = max(f.fields['size_bytes'] for f in allocations)
    if spaces and all(f.evidence == Evidence.PROVEN and f.fields.get('headroom_bytes') is not None for f in spaces):
        values['minimum_headroom_bytes'] = min(f.fields['headroom_bytes'] for f in spaces)
        values['maximum_space_utilization'] = max(f.fields['usage_bytes']/f.fields['limit_bytes'] for f in spaces)
    return [Fact('METRIC', {key:value,'unit':'fraction' if key.endswith('utilization') else 'bytes'}, Evidence.PROVEN)
            for key,value in values.items()] or [Fact('METRIC', {'reason':'compiler resource limits unavailable'}, Evidence.UNAVAILABLE)]


@register('pmu_activity', 'Per-pipeline PMU activity fractions; overlapping pipelines never sum to wall time.', RunIdParams)
def pmu_activity(conn, params):
    complete = common.one(conn, "SELECT value FROM diagnostic_metric WHERE run_id=? AND metric='pmu_coverage_complete'", [params.run_id])
    if complete and complete[0] != 1:
        return [Fact('METRIC', {'reason': 'PMU dispatch coverage incomplete'}, Evidence.UNAVAILABLE)]
    rows = common.q(conn,'SELECT sample_seq,counter,value,total_cycles FROM pmu_counter WHERE run_id=?',[params.run_id])
    samples = {r[0] for r in rows}
    values = {}
    for pipe in ('vec','cube','scalar','mte1','mte2','mte3'):
        matched = [r for r in rows if r[1] in (pipe+'_busy_cycles',pipe+'_cycles')]
        if matched and len(matched)==len(samples) and all(r[3] is not None for r in matched):
            total = sum(r[3] for r in matched)
            if total > 0:
                values[pipe+'_cycle_fraction'] = sum(r[2] for r in matched)/total
    return [Fact('METRIC',{key:value,'unit':'fraction'},Evidence.PROVEN) for key,value in values.items()] or [
        Fact('METRIC',{'reason':'complete pipeline counters unavailable'},Evidence.UNAVAILABLE)]


@register('pipeline_stages', 'Compiler pipeline inclusive and self time; nested child durations are not added twice.', RunIdParams)
def pipeline_stages(conn, params):
    import json
    import math
    from profile_db.runtime_adapter.storage import resolve
    exists = common.one(conn,"SELECT 1 FROM artifact WHERE run_id=? AND kind='pipeline_profile'",[params.run_id])
    if not exists:
        return [Fact('STAGE',{'reason':'pipeline stages unavailable'},Evidence.UNAVAILABLE)]
    raw = json.loads(resolve(conn,params.run_id,'pipeline_profile').read_text())
    facts=[]
    def visit(stages,parent=()):
        for index,stage in enumerate(stages):
            children=stage.get('children',[])
            inclusive=stage['seconds']
            direct=sum(child['seconds'] for child in children)
            valid=all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (inclusive,direct))
            consistent=valid and direct<=inclusive+max(1e-9,inclusive*1e-6)
            position=parent+(index,)
            facts.append(Fact('STAGE',{'position':list(position),'name':stage['name'],
                'inclusive_seconds':inclusive,'self_seconds':max(0,inclusive-direct) if consistent else None,
                'child_count':len(children),'self_basis':'inclusive_minus_direct_children',
                'consistent':consistent},Evidence.PROVEN if consistent else Evidence.UNAVAILABLE))
            visit(children,position)
    visit(raw['stages'])
    return facts


@register('host_cost_summary', 'First and later recorded host invocation costs by process; no synthetic host-device clock join.', RunIdParams)
def host_cost_summary(conn, params):
    import math
    import statistics
    from profile_db.runtime_adapter.storage import read_snapshot
    doc=read_snapshot(conn,params.run_id,'host')
    if doc['status']!='ok':
        return [Fact('METRIC',{'reason':'host analysis unavailable'},Evidence.UNAVAILABLE)]
    payload=doc['payload']
    # Prewarm/build-only invocations have no recorded run costs. They are not
    # the first device execution and must not shift that execution to "later".
    rows=[r for r in payload.get('rounds',[]) if any(
        r.get(key) is not None for key in ('host_us','device_us','effective_us'))]
    if not rows:
        return [Fact('METRIC',{'reason':'recorded host invocations unavailable'},Evidence.UNAVAILABLE)]
    values={}
    first,later=[],[]
    for pid in sorted({r['pid'] for r in rows}):
        process=sorted([r for r in rows if r['pid']==pid],key=lambda r:r['inv'])
        if process:
            first.append(process[0])
            later.extend(process[1:])
    for label,group in (('first',first),('later',later)):
        for key in ('host_us','device_us','effective_us'):
            valid=[r[key] for r in group if type(r.get(key)) in (int,float) and math.isfinite(r[key]) and r[key]>=0]
            if group and len(valid)==len(group):
                values[label+'_'+key]=statistics.median(valid)
        values[label+'_invocations']=len(group)
    return [Fact('METRIC',{key:value,'unit':'count' if key.endswith('invocations') else 'us'},Evidence.PROVEN)
            for key,value in values.items()]
