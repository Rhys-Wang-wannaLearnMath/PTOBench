"""Join the SDK's graph/timing passes by explicit dispatch identity.

Some runtime versions keep local dN counters across prepared launches, while
the SDK assumes those counters reset. Observe the two existing blocking passes
and bind their artifacts before the SDK converts them. No execution, state
reset, configuration, or clean measurement is added or changed.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import re


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('dispatch pairing: artifact missing or symlink: '+str(path))
    return path.read_bytes()


def _captures(root, name):
    paths = sorted(root.glob('rank*/d*/'+name))
    for path in paths:
        if (not re.fullmatch(r'rank[0-9]+',path.parent.parent.name)
                or not re.fullmatch(r'd[0-9]+',path.parent.name)
                or path.parent.is_symlink() or path.parent.parent.is_symlink()):
            raise ValueError('dispatch pairing: invalid capture directory')
    return paths


def _identity(directory):
    identity = json.loads(_read(directory/'dispatch_identity.json'))
    if (not isinstance(identity,dict) or type(identity.get('schema_version')) is not int
            or identity['schema_version']!=1):
        raise ValueError('dispatch pairing: unsupported identity schema')
    for field in ('run_id','task_slot','group_index','group_size','chip_rank','local_capture_index'):
        if type(identity.get(field)) is not int or identity[field]<0:
            raise ValueError('dispatch pairing: invalid identity '+field)
    if (identity['run_id']==0 or identity['group_size']==0
            or identity['group_index']>=identity['group_size']
            or identity['chip_rank']!=int(directory.parent.name[4:])
            or identity['local_capture_index']!=int(directory.name[1:])
            or not isinstance(identity.get('callable_digest'),str)
            or not re.fullmatch(r'[0-9a-f]{64}',identity['callable_digest'])):
        raise ValueError('dispatch pairing: invalid identity or capture path')
    return identity


def _key(identity):
    # Runtime run identifiers can restart between the graph and timing passes.
    # A task slot belongs to its declared rank, group member and callable.
    return tuple(identity[k] for k in
        ('task_slot','chip_rank','group_index','group_size','callable_digest'))


def snapshot(root):
    """Retain graph bytes immediately after the blocking graph pass."""
    root = Path(root)
    return [(path.parent, _read(path), _identity(path.parent)
             if (path.parent/'dispatch_identity.json').exists() else None)
            for path in _captures(root,'deps.json')]


def join(root, graphs):
    """Match both SDK passes by dispatch identity, then colocate their files."""
    root = Path(root)
    records = _captures(root,'chip_swimlane_records.json')
    if not records:
        raise ValueError('dispatch pairing: timing captures missing')
    if not graphs or len(graphs)!=len(records):
        raise ValueError('dispatch pairing: graph/timing dispatch coverage mismatch')
    graph_map = {}
    graph_runs = set()
    for directory,data,identity in graphs:
        if identity is None:
            raise ValueError('dispatch pairing: graph identity missing')
        if _read(directory/'deps.json')!=data:
            raise ValueError('dispatch pairing: graph changed between passes')
        key = _key(identity)
        if key in graph_map:
            raise ValueError('dispatch pairing: ambiguous graph dispatch')
        graph_map[key] = (directory,data,identity)
        graph_runs.add(identity['run_id'])
    timing_runs = set()
    seen = set()
    plan = []
    for path in records:
        identity = _identity(path.parent)
        key = _key(identity)
        if key not in graph_map or key in seen:
            raise ValueError('dispatch pairing: unmatched or ambiguous timing dispatch')
        seen.add(key)
        timing_runs.add(identity['run_id'])
        directory,data,graph_identity = graph_map[key]
        graph_path = path.parent/'deps.json'
        if graph_path.is_symlink() or (graph_path.exists() and _read(graph_path)!=data):
            raise ValueError('dispatch pairing: conflicting timing dependency graph')
        _read(path)
        plan.append((path.parent,data))
    if len(graph_runs)!=1 or len(timing_runs)!=1:
        raise ValueError('dispatch pairing: ambiguous pass run identities')
    if seen!=set(graph_map):
        raise ValueError('dispatch pairing: unused graph dispatch')
    groups = {}
    for _,_,identity in graph_map.values():
        groups.setdefault(identity['task_slot'],[]).append(identity)
    for members in groups.values():
        if (len({m['group_size'] for m in members})!=1
                or len({m['callable_digest'] for m in members})!=1
                or sorted(m['group_index'] for m in members)!=list(range(members[0]['group_size']))):
            raise ValueError('dispatch pairing: incomplete or conflicting dispatch group')
    import file_tree
    for directory,data in plan:
        if not (directory/'deps.json').exists():
            file_tree.atomic_bytes(directory/'deps.json',data)
    return len(plan)


@contextmanager
def capture(*,enabled):
    if not enabled:
        yield
        return
    from pypto.runtime import distributed_runner as runner
    original = getattr(runner,'_run_l3_swimlane_two_pass',None)
    if original is None:
        # SDKs without this L3 protocol keep their original capture path.
        # The evaluator still requires a complete graph for every record.
        yield
        return

    def paired(dc,config,dfx_base,run_pass):
        import golden.runner as golden_runner
        graphs = None
        calls = 0
        def observed(pass_config):
            nonlocal graphs,calls
            if (calls>=2 or bool(pass_config.enable_dep_gen)!=(calls==0)
                    or (calls==0 and pass_config.enable_chip_swimlane!=0)
                    or (calls==1 and pass_config.enable_chip_swimlane!=4)):
                raise ValueError('dispatch pairing: unsupported SDK two-pass protocol')
            if calls == 1 and golden_runner._l3_pass_observer is not None:
                golden_runner._l3_pass_observer()
            value = run_pass(pass_config)
            if calls==0:
                graphs = snapshot(dfx_base)
            calls += 1
            return value
        result = original(dc,config,dfx_base,observed)
        if calls!=2:
            raise ValueError('dispatch pairing: incomplete SDK two-pass protocol')
        join(dfx_base,graphs)
        return result

    runner._run_l3_swimlane_two_pass = paired
    try:
        yield
    finally:
        runner._run_l3_swimlane_two_pass = original
