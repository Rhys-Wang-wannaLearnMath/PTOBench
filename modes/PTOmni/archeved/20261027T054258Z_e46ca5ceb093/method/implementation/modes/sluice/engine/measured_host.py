"""Project saved native BenchmarkStats; no STRACE parsing or phase timing."""
import json
import math
from statistics import median

from .evidence_guard import opaque

VERSION = 'native_host_v2'


def analyze(adapter, handle):
    from profile_db.runtime_adapter.storage import resolve
    conn, run = adapter.db.connection, handle.db_run_id
    resolve(conn, run, 'clean_result')
    stored = conn.execute('SELECT CAST(document AS VARCHAR) FROM clean_measurement WHERE run_id=?', [run]).fetchone()
    doc = json.loads(stored[0]) if stored else {}
    native = doc.get('native_benchmark_stats')
    if not isinstance(native, dict):
        raise ValueError('clean_native_stats_not_recorded')
    if (native.get('schema_version') != 1 or native.get('origin') != 'native_benchmark_stats'
            or native.get('unit') != 'us' or native.get('warmup_excluded') is not True):
        raise ValueError('clean_native_stats_invalid')
    rounds = doc['measurement_config']['actual_rounds']
    series = native.get('per_round', {})

    def valid(values, *, required=False):
        if values is None or values == []:
            if required:
                raise ValueError('clean_native_host_not_recorded')
            return None
        if (not isinstance(values, list) or len(values) != rounds
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values)):
            raise ValueError('clean_native_stats_invalid')
        return values

    if not isinstance(series, dict):
        raise ValueError('clean_native_stats_invalid')
    host = valid(series.get('host'), required=True)
    effective = valid(series.get('effective'), required=True)
    if effective != doc['raw_samples_us']:
        raise ValueError('clean_native_samples_mismatch')
    values = {m: valid(series.get(m)) for m in ('host', 'device', 'effective', 'union')}
    rows = [dict(scope='round', round=i, **{m+'_us': v[i] if v else None for m, v in values.items()})
            for i in range(rounds)]
    for kind in ('per_rank', 'per_dispatch'):
        members = native.get(kind, [])
        if not isinstance(members, list):
            raise ValueError('clean_native_stats_invalid')
        seen = set()
        for member in members:
            if not isinstance(member, dict) or type(member.get('pid')) is not int or member['pid'] <= 0:
                raise ValueError('clean_native_stats_invalid')
            if kind == 'per_dispatch' and (type(member.get('slot')) is not int or member['slot'] < 0):
                raise ValueError('clean_native_stats_invalid')
            identity = (member['pid'], member.get('slot'))
            if identity in seen:
                raise ValueError('clean_native_stats_invalid')
            seen.add(identity)
            metrics = {m: valid(member.get(m)) for m in ('host', 'device', 'effective')}
            ref = opaque('evidence', (handle.ref, 'host_process', member['pid']))
            rows.extend(dict(scope='process' if kind == 'per_rank' else 'dispatch', process_ref=ref,
                             round=i, **({'slot': member['slot']} if kind == 'per_dispatch' else {}),
                             **{m+'_us': v[i] if v else None for m, v in metrics.items()}) for i in range(rounds))
    return {'rows': rows, 'metrics': {'measured_rounds': rounds,
            'median_host_us': median(host), 'maximum_host_us': max(host)},
            'aggregation': 'native_per_round_host', 'warmup_excluded': True,
            'origin': 'native_benchmark_stats', 'interpretation': 'native_benchmark_timings',
            'union_basis': 'native_cross_rank_host_window', 'end_to_end': False,
            'parser_version': VERSION}
