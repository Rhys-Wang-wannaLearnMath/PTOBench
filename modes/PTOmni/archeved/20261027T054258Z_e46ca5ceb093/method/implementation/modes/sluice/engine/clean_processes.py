"""Reconcile retained clean timing records with the exact scored samples.

Uses the owned offline parser, never the installed collector or another launch.
Process references are capture-local. Optional startup records bind them to
runtime-visible devices and the archived local device order.
"""
from collections import defaultdict
import json
import math
from statistics import median

from .contracts import digest
from .evidence_guard import opaque
from .profile_adapter import read_json, within

VERSION = 'clean_processes_v2'
_ROOT = 'chip.run'
_DEVICE = _ROOT + '.runner_run.device_wall'
_ORCH, _SCHED = _DEVICE + '.orch', _DEVICE + '.sched'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _contract(adapter, handle):
    """Old logs can use the original public contract only if its hash matches.

    New attachments archive these files alongside the log. Never rewrite old
    identity or infer expected process coverage from surviving log processes.
    """
    from profile_db.runtime_adapter.storage import resolve
    names = ('case.json', 'config.py', 'comparator.json')
    kinds = ('clean_public_case', 'clean_public_config', 'clean_public_comparator')
    conn, run = adapter.db.connection, handle.db_run_id
    present = {r[0] for r in conn.execute(
        'SELECT kind FROM artifact WHERE run_id=?', [run]).fetchall()}
    if present.intersection(kinds):
        _require(set(kinds) <= present, 'clean_contract_incomplete')
        paths = [resolve(conn, run, kind) for kind in kinds]
    else:
        root = adapter.context.public_context_root(adapter.context.workspace)
        paths = [within(root / name, root) for name in names]
    _require(digest([p.read_text().replace('\r\n', '\n') for p in paths]) ==
             handle.identity.shape_dtype_contract_ref, 'clean_contract_changed')
    case = read_json(paths[0])
    _require(case.get('case_id') == handle.identity.case_id, 'clean_contract_changed')
    identity_runtime = (handle.identity.device or {}).get('runtime') or {}
    runtime = {**identity_runtime,
               **(case.get('runtime') or {})}
    ids = runtime.get('device_ids')
    if ids is None:
        return 1, None
    _require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids)
             and len(set(ids)) == len(ids), 'clean_process_declaration_invalid')
    _require('device_ids' not in identity_runtime or identity_runtime['device_ids']==ids,
             'clean_process_declaration_changed')
    return len(ids), ids


def analyze(adapter, handle):
    from .profile_batches import validate_original
    validate_original(adapter.db, handle)
    from profile_db._offline_tools.strace_timing import (
        _DROP_SUMMARY_RE, count_record_heads, group_invocations, invocation_spans,
        parse_drop_summaries, parse_spans)
    from profile_db.runtime_adapter.storage import resolve
    conn, run = adapter.db.connection, handle.db_run_id
    original = read_json(resolve(conn, run, 'clean_result'))
    row = conn.execute('SELECT CAST(document AS VARCHAR) FROM clean_measurement WHERE run_id=?',
                       [run]).fetchone()
    _require(row is not None, 'clean_measurement_missing')
    measurement = json.loads(row[0])
    receipt = measurement['receipt']
    _require(receipt.get('status') == 'PASS' and receipt.get('correctness') == 'PASS'
             and receipt.get('integrity_passed') is True
             and receipt.get('event_seq') == handle.event_seq, 'clean_receipt_invalid')
    _require(original.get('profiling_mode') == 'clean' and original.get('verdict') == 'PASS'
             and original.get('official_score') is True and original.get('diagnostic_only') is False,
             'clean_measurement_invalid')
    runs = original['runs']
    _require(len(runs) == 1 and runs[0].get('passed') is True
             and runs[0].get('fixture_id') == handle.identity.fixture_id, 'clean_measurement_invalid')
    config = measurement['measurement_config']
    samples = measurement['raw_samples_us']
    _require(runs[0]['measurement_config'] == config
             and runs[0]['samples_effective_us'] == samples
             and runs[0]['raw_samples_effective_us'] == samples, 'clean_measurement_changed')
    rounds, warmup = config.get('actual_rounds'), config.get('actual_warmup')
    _require(type(rounds) is int and rounds > 0 and type(warmup) is int and warmup >= 0
             and config.get('metric') == 'effective' and config.get('unit') == 'us'
             and config.get('benchmark_enabled') is True and len(samples) == rounds
             and all(type(s) in (int, float) and math.isfinite(s) and s > 0 for s in samples),
             'clean_measurement_invalid')
    stored = conn.execute('SELECT round,effective_us FROM bench_sample WHERE run_id=? ORDER BY round',
                          [run]).fetchall()
    _require(stored == list(enumerate(samples)), 'clean_measurement_changed')
    expected,device_ids = _contract(adapter, handle)
    kinds = [r[0] for r in conn.execute(
        "SELECT kind FROM artifact WHERE run_id=? AND kind LIKE 'host_log:%'", [run]).fetchall()]
    _require(kinds == ['host_log:host.benchmark_000.log'], 'clean_process_log_unavailable')
    log = resolve(conn, run, kinds[0])
    text = log.read_text()
    # Two-pass swimlane streams need a separate timing-pass contract; clean
    # streams accepted here have exactly one uninterrupted benchmark pass.
    _require('l3_swimlane_pass=' not in text,
             'clean_timing_pass_unsupported')
    lines = text.splitlines()
    spans = list(parse_spans(lines))
    drops = parse_drop_summaries(lines)
    drop_records = list(_DROP_SUMMARY_RE.finditer(text))
    _require(count_record_heads(lines) == len(spans) and bool(spans)
             and not any(any(v > 0 for v in d.values()) for d in drops.values())
             and len(drop_records) == text.count('[HOSTLOG_DROPS]')
             and all(m['v'] == '1' for m in drop_records), 'clean_process_log_incomplete')
    # New captures anchor the benchmark parser arguments too. Historical logs
    # use independently anchored result config and declared process coverage.
    present = conn.execute("SELECT 1 FROM artifact WHERE run_id=? AND kind='clean_host_manifest'",
                           [run]).fetchone()
    distributed = expected > 1
    if present:
        import hashlib
        manifest = read_json(resolve(conn, run, 'clean_host_manifest'))
        captures = manifest.get('captures', [])
        _require(manifest.get('schema_version') == 1
                 and manifest.get('origin') == 'existing_benchmark_parser_input'
                 and manifest.get('additional_dispatches') == 0
                 and manifest.get('includes_warmup') is True
                 and manifest.get('cold_compile_certified') is False and len(captures) == 1,
                 'clean_host_manifest_invalid')
        item = captures[0]
        _require(item.get('file') == 'host.benchmark_000.log', 'clean_host_manifest_invalid')
        _require(item.get('sha256') == hashlib.sha256(log.read_bytes()).hexdigest()
                 and item.get('rounds') == rounds and item.get('warmup') == warmup
                 and type(item.get('distributed')) is bool
                 and (expected == 1 or item['distributed']), 'clean_host_manifest_invalid')
        distributed = item['distributed']
    groups = group_invocations(invocation_spans(spans))
    by_pid = defaultdict(list)
    for invocation in groups:
        canonical = [s for s in invocation.spans if s.name in (_ROOT, _DEVICE, _ORCH, _SCHED)]
        if not canonical:
            continue
        names = {s.name for s in canonical}
        _require(len(canonical) == len(names), 'clean_process_duplicate_marker')
        _require(len({s.hid for s in canonical}) == 1, 'clean_process_identity_ambiguous')
        root = next((s for s in canonical if s.name == _ROOT and s.depth == 0), None)
        _require(root is not None, 'clean_process_root_missing')
        by_pid[invocation.pid].append((invocation.inv, canonical))
    # Host orchestration may have its own chip.run roots without device spans.
    by_pid = {pid: invs for pid, invs in by_pid.items()
              if any(any(s.name == _DEVICE for s in markers) for _, markers in invs)}
    _require(len(by_pid) == expected, 'clean_process_coverage_incomplete')
    per_process, device_process, counts = {}, {}, {}
    launches = rounds + warmup
    for pid, invocations in sorted(by_pid.items()):
        invocations.sort()
        _require(len(invocations) % launches == 0, 'clean_process_rounds_ambiguous')
        dispatches = len(invocations) // launches
        _require(dispatches > 0 and (distributed or dispatches == 1),
                 'clean_process_rounds_ambiguous')
        _require(all(b[0] == a[0] + 1 for a, b in zip(invocations, invocations[1:])),
                 'clean_process_invocations_missing')
        effective, device = [], []
        for _, markers in invocations:
            by = {s.name: s for s in markers}
            _require({_ROOT, _DEVICE, _ORCH, _SCHED} <= set(by), 'clean_process_marker_missing')
            _require(all(s.ts >= 0 and s.dur > 0 for s in markers)
                     and not by[_ROOT].is_device
                     and all(by[name].is_device for name in (_DEVICE, _ORCH, _SCHED)),
                     'clean_process_clock_invalid')
            windows = [by[_ORCH], by[_SCHED]]
            _require(by[_DEVICE].ts <= min(s.ts for s in windows)
                     and max(s.ts + s.dur for s in windows) <= by[_DEVICE].ts + by[_DEVICE].dur,
                     'clean_process_window_invalid')
            effective.append((max(s.ts + s.dur for s in windows) - min(s.ts for s in windows)) / 1000)
            device.append(by[_DEVICE].dur / 1000)
        per_process[pid] = [sum(effective[k*dispatches:(k+1)*dispatches])
                            for k in range(warmup, launches)]
        device_process[pid] = [sum(device[k*dispatches:(k+1)*dispatches])
                               for k in range(warmup, launches)]
        counts[pid] = dispatches
    reconstructed = [max(v[k] for v in per_process.values()) for k in range(rounds)]
    _require(all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
                 for a, b in zip(samples, reconstructed)), 'clean_process_samples_mismatch')
    from .process_mapping import BASIS, bind
    mapping,mapping_reason = bind(lines,device_ids,per_process)
    rows = []
    spreads, ratios = [], []
    for k, maximum in enumerate(reconstructed):
        minimum = min(v[k] for v in per_process.values())
        spreads.append(maximum - minimum)
        ratios.append(maximum / minimum)
        for pid, values in per_process.items():
            rows.append({'process_ref': opaque('evidence', (handle.ref, 'host_process', pid)),
                         'round': k, 'effective_us': values[k], 'device_us': device_process[pid][k],
                         'dispatch_count': counts[pid], 'round_max_fraction': values[k] / maximum,
                         'rank':mapping.get(pid,{}).get('rank'),'device':mapping.get(pid,{}).get('device')})
    metrics = {'process_count': expected, 'measured_rounds': rounds,
               'median_process_spread_us': median(spreads), 'maximum_process_spread_us': max(spreads),
               'median_process_ratio': median(ratios)}
    return {'metrics': metrics, 'rows': rows, 'aggregation': 'max_of_process_dispatch_sums',
            'warmup_excluded': True, 'rank_mapping_verified': bool(mapping),
            'rank_mapping_basis': BASIS if mapping else None,'rank_mapping_reason':mapping_reason,
            'rank_mapping_scope':'measured_processes_local_runtime_device_order',
            'communication_causality_verified': False, 'parser_version': VERSION}
