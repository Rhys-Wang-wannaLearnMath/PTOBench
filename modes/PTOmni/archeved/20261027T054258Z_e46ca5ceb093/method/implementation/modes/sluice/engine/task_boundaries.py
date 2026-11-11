"""Capture-local tensor views for source optimization, never measured copy traffic.

Read only hash-verified registered originals. No buffer address or raw identity
enters the public projection; unrelated creator/control edges stay explicit.
"""
from collections import Counter
import math

from .evidence_guard import opaque
from .profile_adapter import read_json

VERSION = 'task_boundaries_v2'
_BYTES = {'BOOL':1, 'INT8':1, 'UINT8':1, 'INT16':2, 'UINT16':2,
          'INT32':4, 'UINT32':4, 'INT64':8, 'UINT64':8,
          'FLOAT16':2, 'BFLOAT16':2, 'FLOAT32':4, 'FLOAT64':8}


def task_ref(profile_ref, identity):
    from profile_db.task_ids import normalize_task_id
    return opaque('evidence', (profile_ref, 'task', normalize_task_id(identity).canonical))


def _integer(value):
    if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 20:
        value = int(value)
    if type(value) is not int or not 0 <= value <= 2**63-1:
        raise ValueError('invalid_integer')
    return value


def _view(edge, prefix, dtype, tensor):
    shape, strides = edge.get(prefix+'shape'), edge.get(prefix+'strides')
    if not isinstance(shape, list) or not isinstance(strides, list):
        raise ValueError('view_missing')
    if len(shape) != len(strides) or len(shape) > 32:
        raise ValueError('view_rank_invalid')
    shape, strides = list(map(_integer, shape)), list(map(_integer, strides))
    offset = _integer(edge.get(prefix+'start_offset'))
    if not isinstance(dtype,str) or dtype not in _BYTES:
        raise ValueError('dtype_unavailable')
    if tensor and tensor.get('dtype') != dtype:
        raise ValueError('dtype_conflict')
    count = math.prod(shape)
    # Positive/zero strides only; zero-size views have no addressed elements.
    end = offset + sum((n-1)*s for n,s in zip(shape,strides)) if count else offset
    bounds = None
    if tensor and 'buffer_numel' in tensor:
        size = _integer(tensor['buffer_numel'])
        if (count and end >= size) or (not count and offset > size):
            raise ValueError('view_out_of_bounds')
        bounds = True
    stride, contiguous = 1, True
    for n,s in reversed(list(zip(shape,strides))):
        if n > 1 and s != stride:
            contiguous = False
        stride *= n
    return {'dtype':dtype, 'shape':shape, 'strides':strides,
            'offset_elements':offset, 'logical_bytes':count*_BYTES[dtype],
            'contiguous':contiguous if count else True, 'storage_bounds_verified':bounds}


def analyze(adapter, handle):
    from .profile_batches import validate_original
    from profile_db.runtime_adapter.storage import resolve
    from profile_db.task_ids import normalize_task_id
    validate_original(adapter.db, handle)
    raw = read_json(resolve(adapter.db.connection, handle.db_run_id, 'deps'))
    tasks, edges = raw.get('tasks'), raw.get('edges')
    if not isinstance(tasks,list) or not isinstance(edges,list):
        raise ValueError('boundary_graph_invalid')
    canonical = lambda value: normalize_task_id(value).canonical
    by_task = {canonical(t['task_id']):t for t in tasks}
    tensors = {}
    table = raw.get('tensors',[])
    if not isinstance(table,list):
        raise ValueError('boundary_tensor_table_invalid')
    for tensor in table:
        if not isinstance(tensor,dict) or 'tensor_id' not in tensor:
            raise ValueError('boundary_tensor_table_invalid')
        identity = str(tensor['tensor_id'])
        # Addresses are deliberately neither compared nor projected.
        safe = {k:tensor[k] for k in ('dtype','buffer_numel') if k in tensor}
        if identity in tensors and tensors[identity] != safe:
            raise ValueError('boundary_tensor_metadata_conflict')
        tensors[identity] = safe
    rows = []
    for edge in edges:
        pred, succ = canonical(edge['pred']), canonical(edge['succ'])
        if succ not in by_task or (edge.get('source')!='creator' and pred not in by_task):
            raise ValueError('boundary_graph_invalid')
        source = edge.get('source')
        source = source if source in ('tensormap','creator','explicit') else 'unknown'
        identity = str(edge['tensor_id']) if edge.get('tensor_id') is not None else None
        tensor = tensors.get(identity)
        views, reasons = {}, {}
        for side in ('producer','consumer'):
            try:
                if identity is None or source not in ('tensormap','creator'):
                    raise ValueError('non_tensor_dependency')
                if side == 'producer':
                    if pred not in by_task or source=='creator':
                        raise ValueError('external_producer')
                    args = by_task[pred].get('args',[])
                    if not isinstance(args,list):
                        raise ValueError('producer_argument_unavailable')
                    matches = [a for a in args if isinstance(a,dict)
                        and str(a.get('tensor_id'))==identity and a.get('type') in ('OUTPUT_EXISTING','INOUT')]
                    if len(matches)!=1:
                        raise ValueError('producer_argument_ambiguous')
                    dtype = matches[0].get('dtype')
                else:
                    dtype = edge.get('consumer_dtype')
                views[side] = _view(edge,side+'_',dtype,tensor)
                reasons[side] = None
            except ValueError as exc:
                views[side], reasons[side] = None, str(exc)
        keys = ('dtype','shape','strides','offset_elements')
        relation = ('identical' if all(views['producer'][k]==views['consumer'][k] for k in keys)
                    else 'different') if all(views.values()) else 'unavailable'
        rows.append({'producer_task_ref':task_ref(handle.ref,pred) if pred in by_task and source!='creator' else None,
                     'consumer_task_ref':task_ref(handle.ref,succ),
                     'tensor_ref':opaque('evidence',(handle.ref,'tensor',identity)) if identity is not None else None,
                     'source':source, 'reported_overlap':edge.get('overlap') if edge.get('overlap') in ('covered','partial','disjoint') else None,
                     'producer_view':views['producer'], 'consumer_view':views['consumer'],
                     'producer_reason':reasons['producer'], 'consumer_reason':reasons['consumer'],
                     'view_relation':relation})
    data = [r for r in rows if r['source'] in ('tensormap','creator') and r['tensor_ref'] is not None]
    counts = Counter(r['view_relation'] for r in data)
    return {'rows':rows, 'metrics':{'dependency_edges':len(rows), 'tensor_edges':len(data),
            'identical_views':counts['identical'], 'different_views':counts['different'],
            'unavailable_views':counts['unavailable']}, 'parser_version':VERSION,
            'interpretation':'recorded_tensor_views', 'measured_copy_traffic':False,
            'copy_requirement_verified':False, 'whole_chain_speed_verified':False}
