"""Bounded feedback from a failed public action's trusted receipt or log.

These reports explain failure; they never change receipts, correctness,
measurement identity or performance verdicts. Parent-owned comparator outcomes
identify failed outputs; fixed compiler log vocabulary remains advisory and
does not establish actual memory use or numerical failure.
"""
import copy
import hashlib
import re
import uuid

from .evidence_guard import encoded,opaque,require_ref,scan

MAX_LOG_BYTES=256*1024
MEMORY=re.compile(r'\b(Mat|Vec|Acc|Left|Right|L0A|L0B|L0C) buffer usage '
                  r'\(([0-9]{1,19}) bytes\) exceeds platform limit \(([0-9]{1,19}) bytes\)')


def comparison_feedback(value, source):
    """Accept only the bounded parent projection attached to the public receipt."""
    expected = {'status': 'verified_failure', 'reason': 'output_comparison_failed', 'phase': 'correctness',
        'interpretation': 'trusted_output_comparison_summary', 'verified_failure': True,
        'correctness_verified': False, 'performance_claim': False, 'source_bound': True, 'input_bound': True}
    allowed = set(expected) | {'program_ref', 'evidence_ref', 'fixture_count', 'checked_fixture_count',
                             'failed_fixture_count', 'fixtures', 'truncated'}
    try:
        if (not isinstance(value, dict) or set(value) != allowed
                or any(type(value.get(k)) is not type(v) or value[k] != v for k, v in expected.items())
                or value.get('program_ref') != opaque('program', source) or len(encoded(value)) > 4096):
            return None
        scan(value)
        require_ref(value['evidence_ref'], 'evidence')
        counts = [value[k] for k in ('fixture_count', 'checked_fixture_count', 'failed_fixture_count')]
        if (any(type(n) is not int for n in counts) or not 1 <= counts[2] <= counts[1] <= counts[0]
                or not isinstance(value['fixtures'], list) or not 1 <= len(value['fixtures']) <= 4
                or type(value['truncated']) is not bool or len(value['fixtures']) > counts[2]):
            return None
        slots = set()
        for item in value['fixtures']:
            if (not isinstance(item, dict) or set(item) - {'fixture_index', 'fixture_slot', 'input_ref',
                    'output_count', 'failed_output_count', 'outputs', 'truncated'}
                    or type(item.get('fixture_index')) is not int
                    or not 0 <= item['fixture_index'] < counts[0] or item['fixture_index'] in slots
                    or type(item.get('output_count')) is not int or type(item.get('failed_output_count')) is not int
                    or not 1 <= item['failed_output_count'] <= item['output_count']
                    or not isinstance(item.get('outputs'), list) or not 1 <= len(item['outputs']) <= 8
                    or len(item['outputs']) > item['failed_output_count'] or type(item.get('truncated')) is not bool):
                return None
            if (item['truncated'] != (len(item['outputs']) < item['failed_output_count'])
                    or ('fixture_slot' in item and not isinstance(item['fixture_slot'], str))):
                return None
            slots.add(item['fixture_index'])
            require_ref(item['input_ref'], 'evidence')
            seen = set()
            for row in item['outputs']:
                if (not isinstance(row, dict) or set(row) - {'output_index', 'name', 'status'}
                        or row.get('status') != 'mismatch'
                        or type(row.get('output_index')) is not int
                        or not 0 <= row['output_index'] < item['output_count'] or row['output_index'] in seen
                        or ('name' in row and not isinstance(row['name'], str))):
                    return None
                seen.add(row['output_index'])
        if value['truncated'] != (len(value['fixtures']) < counts[2] or any(i['truncated'] for i in value['fixtures'])):
            return None
        return copy.deepcopy(value)
    except (ValueError, KeyError, TypeError):
        return None


def reported_failure(context,observation):
    receipt=observation.cli_result
    if (observation.operation!='evaluate' or observation.exit_code in (None,0)
            or observation.parse_status!='valid' or not isinstance(receipt,dict)
            or receipt.get('status')!='FAIL' or receipt.get('integrity_passed') is not True
            or receipt.get('correctness') is not None
            or observation.source_before!=observation.source_after):
        return None
    if 'failure_diagnostic' in receipt:
        return comparison_feedback(receipt['failure_diagnostic'], observation.source_before)
    try:
        uuid.UUID(observation.method_action_id)
        root=context.work_root.resolve()
        path=context.work_root/'actions'/observation.method_action_id/'stderr.log'
        if (str(path.relative_to(context.session_root))!=observation.stderr_path
                or path.is_symlink() or path.parent.is_symlink()
                or not path.resolve().is_relative_to(root)):
            return None
        with path.open('rb') as stream:
            stream.seek(0,2)
            stream.seek(max(0,stream.tell()-MAX_LOG_BYTES))
            retained=stream.read(MAX_LOG_BYTES)
    except (OSError,ValueError,TypeError,AttributeError):
        return None
    matches={(space,int(used),int(limit)) for space,used,limit in
             MEMORY.findall(retained.decode('utf-8',errors='replace'))}
    if len(matches)!=1:
        return None
    space,used,limit=next(iter(matches))
    if not 0<limit<used<=2**63-1:
        return None
    value={'status':'reported','reason':'compiler_memory_limit','phase':'compile',
        'space':space,'usage_bytes':used,'limit_bytes':limit,'excess_bytes':used-limit,
        'interpretation':'failed_action_log_advisory','verified_resource_measurement':False,
        'correctness_verified':False,'performance_claim':False,
        'program_ref':opaque('program',observation.source_before),
        'evidence_ref':opaque('evidence',(observation.method_action_id,hashlib.sha256(retained).hexdigest()))}
    scan(value)
    return value
