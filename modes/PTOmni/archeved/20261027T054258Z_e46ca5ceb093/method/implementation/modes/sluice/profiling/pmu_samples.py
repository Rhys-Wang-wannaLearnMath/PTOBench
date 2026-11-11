"""Compact, lossless public PMU pages, including bounded oversized samples."""
from collections import OrderedDict
import math

VERSION = 'pmu-samples-v1'
COORDINATES = ('sample', 'function', 'core', 'thread', 'core_type', 'event_type', 'total_cycles')


def _samples(db, run_id, first):
    rows = db.connection.execute(
        'SELECT sample_seq,func_id,core_id,thread_id,core_type,event_type,total_cycles,task_id,counter,value '
        'FROM pmu_counter WHERE run_id=? AND sample_seq IN '
        '(SELECT DISTINCT sample_seq FROM pmu_counter WHERE run_id=? AND sample_seq>=? '
        'ORDER BY sample_seq LIMIT 17) ORDER BY sample_seq,pmu_id', [run_id, run_id, first]).fetchall()
    samples = OrderedDict()
    for *coords, task, counter, value in rows:
        sample = coords[0]
        if type(sample) is not int or sample < 0:
            raise ValueError('pmu_sample_invalid')
        record = samples.setdefault(sample, {'coords': coords, 'task': task, 'counters': {}})
        if record['coords'] != coords or record['task'] != task:
            raise ValueError('pmu_sample_conflicting_identity')
        if not isinstance(counter, str) or not counter or counter in record['counters']:
            raise ValueError('pmu_sample_duplicate_or_invalid_counter')
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('pmu_sample_invalid_counter')
        total = coords[-1]
        if total is not None and (type(total) not in (int, float) or not math.isfinite(total) or total < 0):
            raise ValueError('pmu_sample_invalid_total')
        record['counters'][counter] = value
    return list(samples.values())


def details(adapter, handle, cursor, budget_bytes):
    from ..engine.controller import save
    from ..engine.evidence_guard import encoded, opaque, require_ref, scan
    from ..engine.profile_adapter import read_json

    binding = opaque('evidence', encoded([handle.ref, handle.artifact_fingerprint, VERSION]).decode())
    root = adapter.context.work_root / 'profile_cursors'
    first, counter_offset = 0, 0
    if cursor is not None:
        require_ref(cursor, 'evidence')
        doc = read_json(root / (cursor.split(':')[1] + '.json'))
        if (doc.get('binding') != binding or type(doc.get('sample')) is not int or doc['sample'] < 0
                or type(doc.get('counter_offset')) is not int or doc['counter_offset'] < 0):
            raise ValueError('detail_cursor_mismatch')
        first, counter_offset = doc['sample'], doc['counter_offset']
    samples = _samples(adapter.db, handle.db_run_id, first)
    if cursor is not None and (not samples or samples[0]['coords'][0] != first
                               or counter_offset >= len(samples[0]['counters'])):
        raise ValueError('detail_cursor_mismatch')

    def envelope(items, position):
        following = opaque('evidence', (binding, position)) if position is not None else None
        return {'profile_ref': handle.ref, 'view': 'pmu_samples', 'items': items,
                'next_cursor': following, 'truncated': following is not None, 'evidence_ref': binding,
                'interpretation': 'sampled_core_activity', 'time_aggregation': False}

    items, position = [], None
    for index, record in enumerate(samples[:16]):
        fields = dict(zip(COORDINATES, record['coords']))
        counters = list(record['counters'].items())
        start = counter_offset if index == 0 else 0
        sample = fields['sample']
        following_sample = samples[index + 1]['coords'][0] if index + 1 < len(samples) else None

        def fragment(stop):
            item = {'entity_ref': opaque('evidence', (handle.ref, 'pmu_samples', sample)),
                    'status': 'measured', 'fields': {**fields, 'counters': dict(counters[start:stop]),
                        'counter_offset': start, 'counter_count': len(counters)}}
            next_position = ((sample, stop) if stop < len(counters) else
                             (following_sample, 0) if following_sample is not None else None)
            page = envelope([*items, item], next_position)
            # Never silently drop unpublishable names or non-finite values.
            scan(page)
            return item, next_position, page

        lo, hi, best = start + 1, len(counters), None
        while lo <= hi:
            stop = (lo + hi) // 2
            candidate = fragment(stop)
            if len(encoded(candidate[2])) <= budget_bytes:
                best = candidate
                lo = stop + 1
            else:
                hi = stop - 1
        if best is None:
            if not items:
                raise ValueError('evidence_budget_invalid')
            position = (sample, start)
            break
        item, position, _ = best
        items.append(item)
        if len(item['fields']['counters']) + start < len(counters):
            break
    page = envelope(items, position)
    scan(page)
    if len(encoded(page)) > budget_bytes:
        raise ValueError('evidence_budget_invalid')
    if page['next_cursor']:
        save(root / (page['next_cursor'].split(':')[1] + '.json'),
             {'binding': binding, 'sample': position[0], 'counter_offset': position[1]})
    return page
