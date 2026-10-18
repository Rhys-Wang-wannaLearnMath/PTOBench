"""Primitive lifecycle facts around the existing Golden runtime dispatch boundary."""
import contextlib
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path

import common
import file_tree


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    file_tree.atomic_bytes(path, (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode())


@contextlib.contextmanager
def observe_runtime(runtime=None):
    """Observe correctness dispatch entry/return, without instrumenting timed rounds.

    An entered dispatch can fail during setup; only a returned dispatch proves
    execution completed. Keep these distinct, and preserve the original error.
    """
    path = os.environ.get('PTOBENCH_EXECUTION_RECORD')
    if not path:
        yield
        return
    if runtime is not None:
        record = common.read_json(Path(path))
        target = {'platform': getattr(runtime, 'platform', None), 'device': getattr(runtime, 'device_id', None)}
        distributed = getattr(runtime, 'distributed_config', None)
        if distributed is not None:
            target['device_ids'] = list(distributed.device_ids)
        record['effective_target'] = target
        write(Path(path), record)
    import golden.runner as runner
    originals = {}
    def update(stage, **fields):
        record = common.read_json(Path(path))
        record['events'].append({'time': now(), 'stage': stage, **fields})
        write(Path(path), record)
    try:
        for name in ('_dispatch', '_run_l3_resident'):
            original = getattr(runner, name)
            originals[name] = original
            def wrap(fn, boundary):
                @functools.wraps(fn)
                def observed(*args, **kwargs):
                    update('runtime_dispatch_entered', boundary=boundary)
                    try:
                        value = fn(*args, **kwargs)
                    except BaseException as error:
                        update('runtime_dispatch_raised', boundary=boundary, exception_type=type(error).__name__, message=str(error))
                        raise
                    update('runtime_dispatch_returned', boundary=boundary)
                    return value
                return observed
            setattr(runner, name, wrap(original, name))
        yield
    finally:
        for name, original in originals.items():
            setattr(runner, name, original)
