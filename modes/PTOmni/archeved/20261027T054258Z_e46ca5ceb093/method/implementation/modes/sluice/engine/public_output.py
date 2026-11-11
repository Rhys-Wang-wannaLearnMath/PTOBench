"""Project internal controller records at the agent transport boundary."""
from .evidence_guard import opaque

_INTERNAL = {'action_id', 'method_action_id', 'event_seq', 'bench_event_seq', 'event_refs',
             'observation_ref', 'result_path', 'stdout_path', 'stderr_path', 'workspace',
             'fork_workspace', 'snapshot_id', 'revision', 'change_scope'}


def project(value):
    if isinstance(value, dict):
        if 'operation' in value and 'cli_result' in value:
            result = value.get('cli_result') or {}
            return {k: result[k] for k in ('status', 'correctness', 'latency_us', 'remaining_hardware_actions') if k in result}
        return {k: ({'improved': 'decreased', 'regressed': 'increased'}.get(v, v)
                    if k in ('verdict', 'comparison') and isinstance(v, str) else project(v))
                for k, v in value.items() if k not in _INTERNAL}
    if isinstance(value, (tuple, list)):
        return [project(v) for v in value]
    if isinstance(value, str) and value.startswith('sha256:'):
        return opaque('program', value)
    return value
