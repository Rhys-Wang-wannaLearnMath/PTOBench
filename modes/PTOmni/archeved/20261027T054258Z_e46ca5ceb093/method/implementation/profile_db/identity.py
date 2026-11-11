# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal capture identity and structured compatibility reports.

No parser or collection entry point lives here. Missing provenance stays missing.
Legacy runs remain queryable, but cannot pass the strict provenance gate.
"""
from __future__ import annotations
import hashlib
import json

IDENTITY_FIELDS = (
    'case_id', 'fixture_id', 'device', 'software', 'shape', 'dtype',
    'shape_dtype_contract_ref', 'tuning', 'profile_mode',
)


def canonical_identity(identity):
    if identity is None:
        return None, None
    if not isinstance(identity, dict):
        raise ValueError('identity must be an object')
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return json.loads(encoded), hashlib.sha256(encoded.encode()).hexdigest()


def profile_handle(conn, run_id):
    row = conn.execute(
        'SELECT program, swimlane_level, clock_freq_hz, num_cores, core_types, core_to_thread '
        'FROM run WHERE run_id=?', [run_id]).fetchone()
    if row is None:
        return None
    doc = conn.execute('SELECT document FROM profile_identity WHERE run_id=?', [run_id]).fetchone()
    identity = json.loads(doc[0]) if doc else None
    artifacts = conn.execute('SELECT kind, sha256 FROM artifact WHERE run_id=? ORDER BY kind, sha256', [run_id]).fetchall()
    from profile_db.runtime_adapter.storage import read_snapshot
    analyses = {family: read_snapshot(conn, run_id, family) for family in ('scheduler', 'critical_path')}
    return dict(run_id=run_id, identity=identity, program=row[0], capture_level=row[1],
                clock=row[2], core_count=row[3], core_types=json.loads(row[4]) if row[4] is not None else None,
                core_topology=json.loads(row[5]) if row[5] is not None else None, artifact_fingerprint=artifacts,
                analyses=analyses)


def compatibility(conn, run_a, run_b, *, source_pair=None):
    a, b = profile_handle(conn, run_a), profile_handle(conn, run_b)
    if a is None or b is None:
        return dict(compatible=False, reasons=['run_missing'])
    reasons = []
    ia, ib = a['identity'], b['identity']
    if ia is None or ib is None:
        reasons.append('identity_missing')
    else:
        for key in IDENTITY_FIELDS:
            if ia.get(key) is None or ib.get(key) is None:
                reasons.append(key + '_missing')
            elif ia[key] != ib[key]:
                reasons.append(key + '_mismatch')
        for key in ('rank_label','invocation'):
            if (key in ia or key in ib) and ia.get(key) != ib.get(key):
                reasons.append(key+'_mismatch')
        # SLUICE NPU captures predating trusted parent UUID collection remain
        # inspectable. Logical IDs alone cannot certify a cross-action compare.
        if run_a != run_b:
            for identity in (ia, ib):
                device = identity.get('device')
                physical = device.get('physical_identity') if isinstance(device, dict) else None
                if (isinstance(device, dict) and device.get('platform') in ('a2a3', 'a5')
                        and (not isinstance(physical, dict) or physical.get('status') != 'verified')):
                    reasons.append('physical_device_identity_unavailable')
        sources = (ia.get('source_digest'), ib.get('source_digest'))
        programs = (ia.get('program_ref'), ib.get('program_ref'))
        if not all(sources) or not all(programs):
            reasons.append('source_missing')
        elif sources[0] != sources[1] or programs[0] != programs[1]:
            if source_pair is None or tuple(source_pair) != sources:
                reasons.append('source_mismatch')
    from profile_db.schema import DIAGNOSTIC_PARSER_VERSION
    modes = {i.get('profile_mode') for i in (ia, ib) if i}
    if modes & {'pmu', 'compiler', 'clean'}:
        protocols = [conn.execute('SELECT mode, parser_version, signature FROM diagnostic_protocol WHERE run_id=?', [r]).fetchone()
                     for r in (run_a, run_b)]
        if not all(protocols) or protocols[0] != protocols[1]:
            reasons.append('diagnostic_protocol_mismatch')
        elif protocols[0][1] != ('benchmark/1' if modes == {'clean'} else DIAGNOSTIC_PARSER_VERSION):
            reasons.append('analysis_stale')
        if a['program'] != b['program']:
            reasons.append('program_mismatch')
        if modes == {'pmu'}:
            completeness = [conn.execute("SELECT value FROM diagnostic_metric WHERE run_id=? AND metric='pmu_coverage_complete'", [r]).fetchone()
                            for r in (run_a, run_b)]
            if any(c and c[0] != 1 for c in completeness):
                reasons.append('pmu_dispatch_coverage_incomplete')
            coverage = [conn.execute("SELECT value FROM diagnostic_metric WHERE run_id=? AND metric='pmu_coverage_fraction'", [r]).fetchone()
                        for r in (run_a,run_b)]
            samples = [conn.execute("SELECT value FROM diagnostic_metric WHERE run_id=? AND metric='pmu_samples'", [r]).fetchone()
                       for r in (run_a, run_b)]
            if not all(samples) or (not all(c and c[0] == 1 for c in coverage) and samples[0] != samples[1]):
                reasons.append('pmu_sample_count_mismatch')
        return dict(compatible=not reasons, reasons=sorted(set(reasons)))
    from profile_db.schema import CAPTURE_PARSER_VERSION
    for run_id in (run_a, run_b):
        version = conn.execute('SELECT capture_parser_version FROM run WHERE run_id=?', [run_id]).fetchone()[0]
        if version != CAPTURE_PARSER_VERSION:
            reasons.append('capture_repair_required')
    for key in ('program', 'capture_level', 'clock', 'core_count', 'core_types', 'core_topology'):
        if a[key] is None or b[key] is None:
            reasons.append(key + '_missing')
        elif a[key] != b[key]:
            reasons.append(key + '_mismatch')
    from profile_db.runtime_adapter.storage import semantic_identity
    for family in a['analyses']:
        da, db = a['analyses'][family], b['analyses'][family]
        if da.get('status') == 'stale' or db.get('status') == 'stale':
            reasons.append('analysis_stale')
        elif semantic_identity(da) != semantic_identity(db):
            reasons.append('analysis_semantics_mismatch')
    return dict(compatible=not reasons, reasons=sorted(set(reasons)))
