"""Complete registered execution counts, independent of bounded fact prefixes.

Only logical-task and physical-row counts are additive across dispatches. These
counts do not establish exposed time, synchronization causes or performance.
Inputs remain separate; a fixture-batch parent is never one execution.
"""
from dataclasses import asdict

from modes.sluice.engine.evidence_guard import encoded

VERSION = 'execution_work_v1'


def analyze(adapter, handle):
    from profile_db.identity import IDENTITY_FIELDS
    from profile_db.runtime_adapter.storage import resolve
    from modes.sluice.engine.profile_batches import validate_original

    if handle.availability != 'available' or handle.identity.profile_mode != 'runtime':
        raise ValueError('execution_work_unavailable')
    if handle.member_kind == 'fixture':
        raise ValueError('execution_work_requires_fixture')
    if handle.member_refs:
        if (handle.member_kind != 'rank_dispatch' or len(handle.member_slots) != len(handle.member_refs)
                or len(set(handle.member_refs)) != len(handle.member_refs)
                or len(set(handle.member_slots)) != len(handle.member_slots)):
            raise ValueError('execution_work_membership_invalid')
        leaves = [adapter.handles[r] for r in handle.member_refs]
        if (tuple(h.identity.rank_label for h in leaves) != handle.member_slots
                or encoded([h.artifact_fingerprint for h in leaves]) != encoded(handle.artifact_fingerprint)):
            raise ValueError('execution_work_membership_invalid')
    else:
        leaves = [handle]
    if len({h.db_run_id for h in leaves}) != len(leaves):
        raise ValueError('execution_work_duplicate_capture')
    totals = {'logical_tasks': 0, 'physical_rows': 0, 'dispatch_count': len(leaves)}
    ranks = {}
    base = asdict(handle.identity)
    for leaf in leaves:
        if leaf.member_refs or leaf.availability != 'available' or leaf.db_run_id is None:
            raise ValueError('execution_work_member_unavailable')
        identity = asdict(leaf.identity)
        for key in (*IDENTITY_FIELDS, 'capture_level', 'source_digest', 'program_ref', 'build_ref'):
            if key == 'device':
                continue
            if base.get(key) is None or identity.get(key) != base[key]:
                raise ValueError('execution_work_identity_mismatch')
        if leaf.action_id != handle.action_id:
            raise ValueError('execution_work_identity_mismatch')
        device = dict(identity['device'])
        rank = device.pop('rank', 0)
        if type(rank) is not int or rank < 0:
            raise ValueError('execution_work_identity_mismatch')
        if handle.member_refs:
            declared = base['device']
            if any(device.get(k) != v for k, v in declared.items() if k != 'device'):
                raise ValueError('execution_work_identity_mismatch')
        elif device != {k: v for k, v in base['device'].items() if k != 'rank'}:
            raise ValueError('execution_work_identity_mismatch')
        if rank in ranks and ranks[rank]['device'] != device:
            raise ValueError('execution_work_identity_mismatch')
        ranks.setdefault(rank, {'device': device, 'members': []})['members'].append(leaf)
        if not adapter.db.compatibility(leaf.db_run_id, leaf.db_run_id)['compatible']:
            raise ValueError('execution_work_capture_not_current')
        validate_original(adapter.db, leaf)
        # A current SQL count alone does not prove the retained input is intact.
        for kind in ('deps', 'chip_swimlane_records'):
            resolve(adapter.db.connection, leaf.db_run_id, kind)
        result = adapter.db.query('runtime_performance', run_id=leaf.db_run_id, budget_bytes=4096)
        for key in ('logical_tasks', 'physical_rows'):
            matches = [f.fields[key] for f in result.facts if f.rec == 'METRIC'
                       and f.evidence.value in ('measured', 'proven') and key in f.fields]
            if len(matches) != 1 or type(matches[0]) is not int or matches[0] < 0:
                raise ValueError('execution_work_counts_unavailable')
            totals[key] += matches[0]
    if handle.member_refs and sorted(ranks) != list(range(len(ranks))):
        raise ValueError('execution_work_membership_invalid')
    if handle.member_refs:
        dispatches = [sorted(h.identity.invocation for h in r['members']) for r in ranks.values()]
        if any(d != dispatches[0] or len(set(d)) != len(d) for d in dispatches):
            raise ValueError('execution_work_membership_invalid')
    totals['rank_count'] = len(ranks)
    return {'metrics': totals, 'ranks': ranks, 'identity': base,
            'scope': 'registered_fixture_execution' if handle.member_refs or handle.identity.rank_label == 'single'
                     else 'atomic_rank_dispatch',
            'complete_registered_members': True, 'time_aggregation': False,
            'performance_claim': False, 'version': VERSION}


def compare(adapter, before, after, *, source_pair):
    """Allow changed dispatch ordinals/counts while retaining all other gates."""
    from profile_db.identity import IDENTITY_FIELDS
    docs = [analyze(adapter, h) for h in (before, after)]
    if any(d['scope'] != 'registered_fixture_execution' for d in docs):
        raise ValueError('execution_work_requires_complete_execution')
    left, right = docs
    if tuple(d['identity']['source_digest'] for d in docs) != tuple(source_pair):
        raise ValueError('execution_work_identity_mismatch')
    for key in (*IDENTITY_FIELDS, 'capture_level'):
        if key != 'device' and left['identity'][key] != right['identity'][key]:
            raise ValueError('execution_work_identity_mismatch')
    if set(left['ranks']) != set(right['ranks']):
        raise ValueError('execution_work_rank_mismatch')
    for rank, row in left['ranks'].items():
        other = right['ranks'][rank]
        if row['device'] != other['device']:
            raise ValueError('execution_work_rank_mismatch')
        # Every member must share core topology, clock and analysis semantics.
        # Labels/ordinals describe individual launches, which may be fused.
        representative = row['members'][0]
        for doc, members in ((left, row['members']), (right, other['members'])):
            for leaf in members:
                pair = (left['identity']['source_digest'], doc['identity']['source_digest'])
                gate = adapter.db.compatibility(representative.db_run_id, leaf.db_run_id, source_pair=pair)
                permitted = {'rank_label_mismatch', 'invocation_mismatch', 'device_mismatch'}
                if set(gate['reasons']) - permitted:
                    raise ValueError('execution_work_capture_incompatible')
    return {'before': left['metrics'], 'after': right['metrics'],
            'deltas': {k: right['metrics'][k] - v for k, v in left['metrics'].items()},
            'comparison_scope': 'registered_execution_counts_only', 'performance_claim': False}
