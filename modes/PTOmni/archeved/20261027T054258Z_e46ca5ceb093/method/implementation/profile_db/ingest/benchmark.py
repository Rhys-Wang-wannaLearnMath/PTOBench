# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Register successful public clean measurements separately from profiling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from profile_db.errors import IngestError
from profile_db.identity import canonical_identity
from profile_db.measurement_stats import compare as compare_samples, summary
from profile_db.ingest.analysis import register_paths
from profile_db.ingest.writer import next_id
from profile_db.runtime_adapter import digest


def ingest(db, source, *, program, identity, receipt, allowed_root=None, copy=False):
    source = Path(source).absolute()
    root = Path(allowed_root or source.parent).resolve()
    if any(p.is_symlink() for p in (source, *source.parents)) or not source.resolve().is_relative_to(root):
        raise IngestError('benchmark artifact outside allowed root')
    identity, key = canonical_identity(identity)
    if not identity or identity.get('profile_mode') != 'clean':
        raise IngestError('benchmark identity missing or mismatched')
    if (receipt.get('status') != 'PASS' or receipt.get('correctness') != 'PASS'
            or receipt.get('integrity_passed') is not True or type(receipt.get('event_seq')) is not int):
        raise IngestError('benchmark public receipt failed')
    if copy and db.path is None:
        raise IngestError('copy archive mode requires a file-backed database')
    before = digest(source)
    try:
        raw = json.loads(source.read_text())
        if (raw.get('profiling_mode') != 'clean' or raw.get('official_score') is not True
                or raw.get('diagnostic_only') is not False or raw.get('verdict') != 'PASS'
                or raw.get('case_id') != identity.get('case_id')):
            raise ValueError('benchmark result is not a passing clean evaluation')
        runs = [r for r in raw['runs'] if r.get('fixture_id') == identity.get('fixture_id')]
        if len(runs) != 1 or runs[0].get('passed') is not True:
            raise ValueError('benchmark fixture missing or failed')
        run = runs[0]
        config = run['measurement_config']
        if config.get('benchmark_enabled') is not True or config.get('metric') != 'effective' or config.get('unit') != 'us':
            raise ValueError('benchmark measurement configuration missing')
        measured = summary(run['samples_effective_us'])
        if config.get('actual_rounds') != measured['count']:
            raise ValueError('benchmark samples incomplete')
        raw_samples = run.get('raw_samples_effective_us')
        if not isinstance(raw_samples, list) or raw_samples != run['samples_effective_us']:
            raise ValueError('benchmark invalid or missing raw timing slots')
        protocol = {k: config.get(k) for k in ('metric', 'unit', 'benchmark_enabled', 'reset_inputs', 'actual_warmup')}
        if protocol['actual_warmup'] is None or type(protocol['reset_inputs']) is not bool:
            raise ValueError('benchmark warmup/reset policy missing')
    except (KeyError, TypeError, ValueError) as exc:
        raise IngestError(str(exc)) from exc
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    doc = {'schema_version': 1, 'receipt': receipt, 'measurement_config': config,
        'raw_samples_us': raw_samples, 'pairing': run.get('pairing'),
        'cached_reference_median_us': run.get('ref_median_effective_us'),
        'summary': measured}
    if 'native_benchmark_stats' in run:
        doc['native_benchmark_stats'] = run['native_benchmark_stats']
    copied = []
    try:
        with db._writing() as conn:
            row = conn.execute('SELECT run_id FROM run WHERE identity_key=?', [key]).fetchone()
            if row:
                old = conn.execute("SELECT sha256 FROM artifact WHERE run_id=? AND kind='clean_result'", [row[0]]).fetchone()
                stored = conn.execute('SELECT CAST(document AS VARCHAR) FROM clean_measurement WHERE run_id=?', [row[0]]).fetchone()
                if not old or old[0] != before or not stored or json.loads(stored[0]) != doc:
                    raise IngestError('registered benchmark action changed')
                return {'run_id': row[0], 'mode': 'clean', 'samples': measured['count']}
            run_id = next_id(conn, 'run', 'run_id')
            conn.execute('INSERT INTO run(run_id,program,identity_key,bench_min_us,bench_median_us,bench_mean_us,bench_max_us,bench_rounds) VALUES (?,?,?,?,?,?,?,?)',
                [run_id, program, key, measured['min_us'], measured['median_us'], measured['mean_us'], measured['max_us'], measured['count']])
            conn.execute('INSERT INTO profile_identity VALUES (?,?)', [run_id, json.dumps(identity, sort_keys=True)])
            conn.execute('INSERT INTO clean_measurement VALUES (?,?)', [run_id, json.dumps(doc, sort_keys=True)])
            conn.execute('INSERT INTO diagnostic_protocol VALUES (?,?,?,?)', [run_id, 'clean', 'benchmark/1', signature])
            conn.executemany('INSERT INTO bench_sample(run_id,round,effective_us) VALUES (?,?,?)',
                [(run_id, i, value) for i, value in enumerate(measured['samples_us'])])
            register_paths(conn, run_id, {'clean_result': source}, copy=copy, copied=copied)
            if digest(source) != before:
                raise IngestError('benchmark artifact changed during ingest')
    except Exception:
        for path in copied:
            path.unlink(missing_ok=True)
        raise
    return {'run_id': run_id, 'mode': 'clean', 'samples': measured['count']}


def compare(db, before, after, *, source_pair=None):
    gate = db.compatibility(before, after, source_pair=source_pair)
    if not gate['compatible']:
        return {'comparison': 'incomparable', 'reasons': gate['reasons']}
    docs = []
    for run in (before, after):
        row = db.connection.execute('SELECT CAST(document AS VARCHAR) FROM clean_measurement WHERE run_id=?', [run]).fetchone()
        if not row:
            return {'comparison': 'unavailable', 'reasons': ['clean_measurement_missing']}
        # Revalidate the saved original before using its samples.
        from profile_db.runtime_adapter.storage import resolve
        resolve(db.connection, run, 'clean_result')
        docs.append(json.loads(row[0]))
    report = compare_samples(docs[0]['raw_samples_us'], docs[1]['raw_samples_us'])
    report.update(before_run=before, after_run=after, correctness='PASS',
                  baseline_kind='public_evaluation', parser_version='benchmark/2')
    return report
