# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime work and occupancy without converting observations to causality."""
from collections import defaultdict
import math

from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.params import RunIdParams, ProfileDetailParams
from profile_db.query.registry import register


def interval_union(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(end, stop) if end is not None else stop
    return total


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _delay(start, end):
    return end-start if _finite(start) and _finite(end) and end >= start else None


def _occupancy(rows, topology):
    """One calculation for engine summaries and core drill-down, including idle cores."""
    valid = [r for r in rows if _delay(r[2], r[3]) is not None]
    if not valid:
        return {}, []
    t0, t1 = min(r[2] for r in valid), max(r[3] for r in valid)
    span, by_core = t1-t0, defaultdict(list)
    for core, engine, start, end, *_ in valid:
        by_core[(core, engine)].append((start, end))
    complete = (span > 0 and bool(topology) and len(valid) == len(rows) and
                all(type(c) is int and 0 <= c < len(topology) and topology[c] == e
                    for c, e in by_core))
    values, details = {}, []
    tail_start = t0+.9*span
    for core, engine in enumerate(topology):
        if engine not in ('aic', 'aiv'):
            continue
        intervals = by_core[(core, engine)]
        busy = interval_union(intervals) if complete else None
        tail = interval_union([(max(s, tail_start), t) for s, t in intervals if t > tail_start]) if complete else None
        details.append(dict(entity=str(core), core=core, engine=engine, complete=bool(complete),
                            window_start_us=t0, window_end_us=t1, tail_start_us=tail_start,
                            busy_us=busy, occupancy=busy/span if complete else None,
                            tail_occupancy=tail/(t1-tail_start) if complete else None))
    if complete:
        for engine in ('aic', 'aiv'):
            cores = [r for r in details if r['engine'] == engine]
            if cores:
                values[engine+'_busy_core_us'] = sum(r['busy_us'] for r in cores)
                values[engine+'_occupancy'] = sum(r['occupancy'] for r in cores)/len(cores)
                values[engine+'_tail_occupancy'] = sum(r['tail_occupancy'] for r in cores)/len(cores)
    return values, details


@register('runtime_performance', 'Logical work, physical dispatch, critical-path work and interval-union engine occupancy.', RunIdParams)
def runtime_performance(conn, params):
    run = common.one(conn, 'SELECT makespan_us, cpm_us, CAST(core_types AS VARCHAR) FROM run WHERE run_id=?', [params.run_id])
    if run is None:
        return common.run_missing('METRIC', params.run_id)
    rows = common.q(conn, 'SELECT core_index, engine, start_us, end_us, dispatch_us, receive_us, finish_us FROM task_row WHERE run_id=?', [params.run_id])
    count = common.one(conn, 'SELECT count(*) FROM task WHERE run_id=?', [params.run_id])[0]
    values = {'logical_tasks': count, 'physical_rows': len(rows)}
    units = {'logical_tasks':'count', 'physical_rows':'count'}
    valid = [(c,e,s,t,d,r,f) for c,e,s,t,d,r,f in rows
             if all(type(v) in (int,float) and math.isfinite(v) for v in (s,t)) and t >= s]
    occupancy, _ = _occupancy(rows, common.json_cell(run[2]) or [])
    values.update(occupancy)
    units.update({key: 'core_us' if key.endswith('busy_core_us') else 'fraction' for key in occupancy})
    if valid:
        # Match coordinates on each physical row. Independent task extrema
        # would combine different cores and invent a scheduling delay.
        for key,a,b in (('dispatch_to_receive_core_us',4,5),('receive_to_start_core_us',5,2),('end_to_finish_core_us',3,6)):
            if len(valid) == len(rows) and all(type(r[a]) in (int,float) and type(r[b]) in (int,float)
                    and math.isfinite(r[a]) and math.isfinite(r[b]) and r[b]>=r[a] for r in valid):
                values[key] = sum(r[b]-r[a] for r in valid)
                units[key] = 'core_us'
    from profile_db.runtime_adapter.storage import read_snapshot
    snapshot = read_snapshot(conn, params.run_id, 'critical_path')
    if snapshot['status'] == 'ok':
        if run[1] is not None:
            values['cpm_us'], units['cpm_us'] = run[1], 'us'
        # The observed frontier sweep partitions the path into disjoint
        # compute and stall segments. Individual ready-to-start gaps can
        # overlap and must not be summed as exposed path time.
        path = common.q(conn, "SELECT compute_us, stall_us FROM cpm_path WHERE run_id=? AND kind='observed'", [params.run_id])
        for col,key in enumerate(('observed_path_compute_us','observed_path_gap_us')):
            if path and all(type(r[col]) in (int,float) and math.isfinite(r[col]) and r[col]>=0 for r in path):
                values[key],units[key] = sum(r[col] for r in path),'us'
    return [Fact('METRIC', {key:common.us(value) if units[key] in ('us','core_us') else
                            round(value,6) if units[key]=='fraction' else value,'unit':units[key]}, Evidence.PROVEN)
            for key,value in values.items()]


@register('async_poll_summary', 'How much async-completion polling activity was recorded? This is not DMA wait time or complete coverage.', RunIdParams)
def async_poll_summary(conn, params):
    from profile_db.scheduler_observations import analyze
    from profile_db.errors import QueryError
    if common.one(conn, 'SELECT 1 FROM run WHERE run_id=?', [params.run_id]) is None:
        return common.run_missing('METRIC', params.run_id)
    try:
        doc = analyze(conn, params.run_id)
    except QueryError as exc:
        if str(exc) != 'scheduler_phase_capture_absent':
            raise
        return [Fact('SCOPE', {'reason': str(exc)}, Evidence.UNAVAILABLE)]
    units = ('count', 'count', 'lane_us', 'us')
    return [Fact('METRIC', {key: value, 'unit': unit}, Evidence.PROVEN)
            for (key, value), unit in zip(doc['metrics'].items(), units)] + [
                Fact('SCOPE', {k: v for k, v in doc.items() if k not in ('rows', 'metrics')}, Evidence.PROVEN)]


@register('profile_details', 'Stable bounded pages of critical tasks, scheduler phases, PMU samples and compiler allocation evidence.', ProfileDetailParams)
def profile_details(conn, params):
    args = [params.run_id, params.limit+1, params.offset]
    if params.view in ('engine_occupancy', 'dispatch_delays'):
        if params.view == 'engine_occupancy':
            run = common.one(conn, 'SELECT CAST(core_types AS VARCHAR) FROM run WHERE run_id=?', [params.run_id])
            rows = common.q(conn, 'SELECT core_index,engine,start_us,end_us FROM task_row WHERE run_id=?', [params.run_id])
            _, found = _occupancy(rows, common.json_cell(run[0]) or []) if run else ({}, [])
            scope = dict(interpretation='observed_compute_window_occupancy', tail_fraction=.1,
                         includes_configured_idle_cores=True, performance_score=False)
        else:
            fields = ('entity', 'name', 'core', 'engine', 'row_index', 'dispatch_us', 'receive_us', 'start_us', 'end_us', 'finish_us')
            rows = common.q(conn, 'SELECT r.task_id,t.name,r.core_index,r.engine,r.row_index,r.dispatch_us,r.receive_us,r.start_us,r.end_us,r.finish_us '
                'FROM task_row r LEFT JOIN task t ON r.run_id=t.run_id AND r.task_id=t.task_id '
                'WHERE r.run_id=? ORDER BY r.task_id,r.core_index,r.engine,r.row_index LIMIT ? OFFSET ?', args)
            found = [dict(zip(fields, row)) for row in rows]
            for row in found:
                for key, start, end in (('dispatch_to_receive_us','dispatch_us','receive_us'),
                                        ('receive_to_start_us','receive_us','start_us'),
                                        ('end_to_finish_us','end_us','finish_us')):
                    row[key] = _delay(row[start], row[end])
                row['complete'] = all(row[k] is not None for k in ('dispatch_to_receive_us','receive_to_start_us','end_to_finish_us'))
            scope = dict(interpretation='matched_physical_dispatch_coordinates', aggregate_unit='core_us',
                         row_unit='us', causal_wait=False, performance_score=False)
        start = params.offset if params.view == 'engine_occupancy' else 0
        selected = found[start:start+params.limit]
        return [Fact('DETAIL', row, Evidence.PROVEN if row['complete'] else Evidence.UNAVAILABLE) for row in selected] + [
            Fact('PAGE', {'has_more': len(found) > start+params.limit, 'next_offset': params.offset+len(selected)}, Evidence.PROVEN),
            Fact('SCOPE', scope, Evidence.PROVEN)]
    if params.view == 'scheduler_phases':
        from profile_db.scheduler_observations import analyze
        doc = analyze(conn, params.run_id)
        details = [Fact('DETAIL', row, Evidence.MEASURED)
                   for row in doc['rows'][params.offset:params.offset+params.limit]]
        return details + [Fact('PAGE', {'has_more': len(doc['rows']) > params.offset+params.limit,
                                       'next_offset': params.offset+len(details)}, Evidence.PROVEN),
                          Fact('SCOPE', {k: v for k, v in doc.items() if k not in ('rows', 'metrics')}, Evidence.PROVEN)]
    if params.view == 'tasks':
        fields = ('entity','name','engine','wall_us','busy_us','physical_rows','critical')
        rows = common.q(conn, 'SELECT task_id,name,engine,wall_us,busy_us,num_rows,on_cpm_observed FROM task '
            'WHERE run_id=? ORDER BY wall_us DESC NULLS LAST,task_id LIMIT ? OFFSET ?',args)
    elif params.view == 'critical_path':
        from profile_db.runtime_adapter.storage import read_snapshot
        if read_snapshot(conn,params.run_id,'critical_path')['status'] != 'ok':
            return [Fact('DETAIL', {'reason':'critical analysis unavailable'},Evidence.UNAVAILABLE)]
        fields = ('entity','position','compute_us','gap_us','stall_us')
        rows = common.q(conn, "SELECT task_id,seq,compute_us,gap_us,stall_us FROM cpm_path WHERE run_id=? AND kind='observed' ORDER BY seq LIMIT ? OFFSET ?",args)
    elif params.view == 'pmu':
        fields = ('entity','sample','function','core','thread','core_type','event_type','counter','cycles','total_cycles')
        rows = common.q(conn, 'SELECT task_id,sample_seq,func_id,core_id,thread_id,core_type,event_type,counter,value,total_cycles '
            'FROM pmu_counter WHERE run_id=? ORDER BY sample_seq,pmu_id LIMIT ? OFFSET ?',args)
    elif params.view == 'hints':
        from profile_db.query.hints import details
        return details(conn, params)
    elif params.view == 'samples':
        fields = ('entity','sample','effective_us')
        rows = common.q(conn,'SELECT CAST(round AS VARCHAR),round,effective_us FROM bench_sample WHERE run_id=? ORDER BY stratum,round LIMIT ? OFFSET ?',args)
    elif params.view in ('pipeline','host'):
        if params.view=='pipeline':
            from profile_db.query.handlers_modalities import pipeline_stages
            found=pipeline_stages(conn,params)
            details=[Fact('DETAIL',{**f.fields,'entity':str(f.fields.get('position'))},f.evidence)
                     for f in found[params.offset:params.offset+params.limit] if f.rec=='STAGE']
        else:
            from profile_db.runtime_adapter.storage import read_snapshot
            doc=read_snapshot(conn,params.run_id,'host')
            found=doc['payload'].get('rounds',[]) if doc['status']=='ok' else []
            details=[Fact('DETAIL',{**r,'entity':str(r['pid'])+':'+str(r['inv'])},Evidence.MEASURED)
                     for r in found[params.offset:params.offset+params.limit]]
        return details+[Fact('PAGE',{'has_more':len(found)>params.offset+params.limit,'next_offset':params.offset+len(details)},Evidence.PROVEN)]
    else:
        from profile_db.query.handlers_modalities import compiler_resources
        kind = 'RESOURCE' if params.view == 'resources' else 'ALLOCATION'
        found = [f for f in compiler_resources(conn,params) if f.rec == kind]
        found.sort(key=lambda f:(f.fields.get('kernel',''),f.fields.get('space',''),f.fields.get('buffer','')))
        details = [Fact('DETAIL',{**f.fields,'entity':str(f.fields.get('kernel'))+':'+str(f.fields.get('space'))+':'+str(f.fields.get('buffer',''))},f.evidence)
                   for f in found[params.offset:params.offset+params.limit]]
        return details+[Fact('PAGE',{'has_more':len(found)>params.offset+params.limit,'next_offset':params.offset+len(details)},Evidence.PROVEN)]
    details = [Fact('DETAIL',dict(zip(fields,row)),Evidence.MEASURED) for row in rows[:params.limit]]
    return details+[Fact('PAGE',{'has_more':len(rows)>params.limit,'next_offset':params.offset+len(details)},Evidence.PROVEN)]
