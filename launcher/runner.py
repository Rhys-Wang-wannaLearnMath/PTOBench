"""Shared process lifecycle, log draining and S4 automatic finalization."""
from contextlib import contextmanager
import os
import selectors
import signal
import stat
import sys
import time

import common
import file_tree
import integrity
import session
import session_actions as actions
from snapshots import finalize_session
from launcher.preflight import preflight
from launcher import provenance, methods


@contextmanager
def signals():
    state = {'received': None}
    previous = {}
    def receive(signum, frame):
        state['received'] = state['received'] or signum
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, receive)
        yield state
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def save(root, name, value):
    """Record provenance in S1, under the same short lock as Session updates."""
    with session._session_lock(root):
        session._load_locked(root)
        with file_tree.open_directory((root / name).parent, create=True):
            pass
        logical = 'launcher_' + name.replace('.', '_').replace('/', '_')
        manifest = integrity._load(root / 'protection')
        if logical not in manifest['roots']:
            integrity.register_roots(root / 'protection',
                {logical: integrity.ProtectionRoot(root / name, 'records')})
        payload = value.encode() if isinstance(value, str) else session._bytes(value)
        integrity.commit_records(root / 'protection', {logical: payload})


def retain_method_reply(root, reply):
    """Archive participant-supplied records without interpreting their contents."""
    records = reply.get('records', {})
    archives = reply.get('archives', {})
    if not isinstance(records, dict) or not isinstance(archives, dict):
        raise integrity.InfrastructureError('invalid method archive declaration')
    def relative(name):
        from pathlib import PurePosixPath
        path = PurePosixPath(name) if isinstance(name, str) else None
        if (path is None or path.is_absolute() or '..' in path.parts or not path.parts
                or str(path) != name or path.parts[0] not in ('agent', 'method')
                and name not in ('prompt.txt', 'profile.json', 'mode.json', 'backend.json')):
            raise integrity.InfrastructureError('invalid method record path')
        return root / name
    for name, value in records.items():
        relative(name)
        save(root, name, value)
    with session._session_lock(root):
        session._load_locked(root)
        roots = {}
        for logical, name in archives.items():
            session._id(logical, 'Archive ID')
            path = relative(name)
            if name in records or name in ('prompt.txt', 'profile.json', 'mode.json', 'backend.json'):
                raise integrity.InfrastructureError('method archive overlaps protected records')
            roots['participant_' + logical] = integrity.ProtectionRoot(path, 'agent_archive', readonly=True)
        if roots:
            integrity.register_roots(root / 'protection', roots)


class Logs:
    """Spool both streams to disk so slow terminal consumers cannot stall cleanup."""
    def __init__(self, root):
        self.selector = selectors.DefaultSelector()
        self.files = []
        self.outputs = []
        self.combined = self.open(root / 'process.log')
        self.root = root

    @staticmethod
    def open(path):
        with file_tree.open_parent(path) as (fd, name):
            log_fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=fd)
            if not stat.S_ISREG(os.fstat(log_fd).st_mode):
                os.close(log_fd)
                raise integrity.InfrastructureError('process log is not a regular file')
        return os.fdopen(log_fd, 'w+b', buffering=0)

    def attach(self, process):
        for pipe, terminal, name in ((process.stdout, sys.stdout, 'stdout.log'), (process.stderr, sys.stderr, 'stderr.log')):
            log = self.open(self.root / name)
            self.files.append(log)
            os.set_blocking(pipe.fileno(), False)
            self.selector.register(pipe, selectors.EVENT_READ, log)
            try:
                fd = terminal.fileno()
                blocking = os.get_blocking(fd)
                os.set_blocking(fd, False)
                self.outputs.append([fd, blocking, log, 0, False])
            except (AttributeError, OSError, ValueError):
                self.outputs.append([None, None, log, 0, True])

    def drain(self, timeout=.05):
        for key, _ in self.selector.select(timeout):
            try:
                data = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if data:
                key.data.seek(0, os.SEEK_END)
                key.data.write(data)
                self.combined.write(data)
            else:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
        actions._forward(self.outputs)

    def close(self):
        try:
            # Called after owned groups have stopped; bounded even if a detached
            # foreign process retains an FD. Lifecycle never waits for pipe EOF.
            deadline = time.monotonic() + 2
            while self.selector.get_map() and time.monotonic() < deadline:
                self.drain(.01)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and actions._forward(self.outputs):
                time.sleep(.005)
        finally:
            for key in list(self.selector.get_map().values()):
                key.fileobj.close()
            self.selector.close()
            for fd, blocking, *_ in self.outputs:
                if fd is not None:
                    os.set_blocking(fd, blocking)
            for log in self.files + [self.combined]:
                log.close()


def _stop_action(root, active):
    # Only the evaluator identity recorded for this Session is owned here.
    if active and active.get('process'):
        actions._stop_group(active['process'])


def _owned_marker(root, process):
    """Retain a pinned child identity when protected records become unreadable.

    The global marker alone cannot authorize killing a foreign evaluator: its
    live action owner must still belong to this Optimizer's original group.
    """
    if process is None:
        return None
    try:
        marker = session._read(actions._marker_path())
        owner = marker.get('owner')
        child = marker.get('process')
        if (marker.get('schema_version') == 1 and marker.get('session_root') == str(root)
                and owner and child and session.process_identity(owner['pid']) == owner
                and owner['pgid'] == process.ptobench_identity['pgid']
                and session.process_identity(child['pid']) == child and child['pid'] == child['pgid']):
            return {'process': child}
    except Exception:
        pass
    return None


def run(path):
    experiment, method, ready, case = preflight(path)
    metadata = ready.get('optimizer_metadata', {})
    if not isinstance(metadata, dict) or set(metadata) - {'command', 'tool_version'}:
        raise methods.MethodError('invalid participant metadata', 'infrastructure_error')
    optimizer = {**experiment.optimizer, **metadata}
    handle = session.create_session(experiment.case, optimizer, experiment.output, experiment.budget, mode=experiment.mode)
    root = handle.session_root
    record = session.read_session(root)
    environment = {**os.environ, 'PATH': str(common.BENCHMARK_ROOT) + os.pathsep + os.environ.get('PATH', ''),
                   'PTOBENCH_SESSION_ROOT': str(root), 'PTOBENCH_SESSION_ID': record['session_id'],
                   'PTOBENCH_CASE_ID': record['case_id'], 'PTOBENCH_OPTIMIZER_ID': record['optimizer_id'],
                   'PTOBENCH_AGENT_BACKEND': experiment.optimizer['backend'],
                   'PTOBENCH_WORKSPACE': str(handle.initial_workspace)}
    request = {'schema_version': 1, 'experiment': experiment.to_dict(), 'preflight': ready,
               'context': {'session_root': str(root), 'workspace': str(handle.initial_workspace),
                           'session_id': record['session_id'], 'budget': record['budget'],
                           'profiling_modes': record['profiling_modes'],
                           'case': {key: case[key] for key in ('case_id', 'module', 'entry')}}}
    process = None
    logs = None
    active = None
    reason = 'optimizer_exit'
    error = None
    final = None
    code = None
    agent_record = None
    collection_ready = True
    with signals() as state:
        try:
            save(root, 'experiment.json', experiment.to_dict())
            save(root, 'method.json', method)
            save(root, 'method_request.json', request)
            save(root, 'baseline.json', session._read(record['sealed_dir'] + '/ref_performance.json'))
            save(root, 'environment.json', provenance.collect_environment(root, experiment.case))
            prepared = methods.call(method, 'prepare', request, environment=environment)
            retain_method_reply(root, prepared)
            logs = Logs(root)
            process = methods.start(method, root / 'method_request.json', handle.initial_workspace, environment)
            save(root, 'optimizer_process.json', process.ptobench_identity)
            logs.attach(process)
            last_poll = 0
            while True:
                logs.drain()
                now = time.monotonic()
                if now - last_poll >= 1 or state['received'] or process.poll() is not None:
                    record = session.read_session(root)
                    active = record['active_action']
                    last_poll = now
                    if state['received']:
                        reason = 'user_interrupt'
                        break
                    if record['status'] == 'failed':
                        raise integrity.InfrastructureError('Session has an infrastructure failure')
                    if process.poll() is not None:
                        code = process.returncode
                        reason = 'optimizer_exit' if code == 0 else 'optimizer_nonzero_exit'
                        break
                    maximum = record['budget']['max_hardware_actions']
                    if (maximum != session.UNLIMITED_HARDWARE_ACTIONS
                            and record['usage']['hardware_actions'] >= maximum
                            and active is None):
                        reason = 'budget_exhausted'
                        break
            session.close_exploration(root, reason)
        except Exception as failure:
            error = getattr(failure, 'error_code', 'infrastructure_error')
            reason = 'launcher_infrastructure_failure'
        finally:
            # Order matters: prevent new actions, stop/reap Optimizer owners,
            # then stop separate evaluator groups before acquiring hardware lock.
            try:
                session.close_exploration(root, reason)
                active = session.read_session(root)['active_action']
            except Exception:
                error = 'infrastructure_error'
                active = _owned_marker(root, process) or active
            try:
                if process is not None:
                    actions._stop_group(process.ptobench_identity, process)
                    code = process.returncode
            except Exception:
                error = 'infrastructure_error'
                collection_ready = False
            try:
                _stop_action(root, active)
                with session.execution_lock():
                    actions.recover_global_action()
                    actions._recover_unmarked_action(root)
                    # Restore protected entrypoint/tool code before collect executes
                    # in a fresh interpreter, including edits after the last action.
                    available_groups = {value['group'] for value in integrity._load(root / 'protection')['roots'].values()}
                    integrity.check_and_restore(root / 'protection', {'fixed', 'agent_archive'} & available_groups)
                    if error:
                        session.fail_session(root, reason)
            except Exception:
                error = 'infrastructure_error'
                collection_ready = False
            finally:
                if logs is not None:
                    try:
                        logs.close()
                    except Exception:
                        error = 'infrastructure_error'
            try:
                if not collection_ready:
                    raise integrity.InfrastructureError('method cleanup or source restoration failed')
                collected = methods.call(method, 'collect', {**request, 'started': process is not None,
                                                            'close_reason': reason}, environment=environment)
                retain_method_reply(root, collected)
                agent_record = collected.get('recording')
                if agent_record is not None and (not isinstance(agent_record, dict)
                                                or not isinstance(agent_record.get('status'), str)):
                    raise integrity.InfrastructureError('invalid participant recording status')
                save(root, 'method_collection.json', {'schema_version': 1, 'recording': agent_record,
                     'record_paths': list(collected.get('records', {})), 'archives': collected.get('archives', {})})
            except Exception:
                error = 'infrastructure_error'
        if not error:
            final = finalize_session(root, reason).to_dict()
        # Preserve verified performance while reporting missing participant history.
        if not error and agent_record and agent_record['status'] == 'missing':
            error = 'recording_incomplete'
        try:
            save(root, 'launcher_result.json', {'close_reason': reason, 'optimizer_exit_code': code,
                 'signal': state['received'], 'error_code': error, 'agent_recording': agent_record,
                 'final_result': str(root / 'final_result.json') if final else None})
        except Exception:
            error = 'infrastructure_error'
        exit_code = (130 if state['received'] or final and final['error_code'] == 'interrupted'
                     else 5 if error or final and final['error_code'] == 'infrastructure_error'
                     else 0 if final and final['status'] == 'completed' else 1)
    return {'session_root': str(root), 'final_result_path': str(root / 'final_result.json') if final else None,
            'status': final['status'] if final and not error else 'failed', 'error_code': error or (final or {}).get('error_code'),
            'close_reason': reason, 'optimizer_exit_code': code, 'agent_recording': agent_record}, exit_code


def status(root):
    with session._session_lock(root) as normalized:
        record, _ = session._load_locked(normalized)
        manifest = integrity._load(normalized / 'protection')
    result = {key: record[key] for key in ('session_id', 'optimizer_id', 'case_id', 'status',
              'exploration_closed', 'close_reason', 'budget', 'usage', 'active_action')}
    result['remaining_hardware_actions'] = actions._remaining(record)
    result['mode'] = record.get('mode')
    result['session_root'] = record['session_root']
    has_final = manifest['roots']['final_result']['entries']['']['kind'] == 'file'
    result['final_result_path'] = record['session_root'] + '/final_result.json' if has_final else None
    if 'launcher_method_collection_json' in manifest['roots']:
        result['agent_recording'] = session._read(normalized / 'method_collection.json').get('recording')
    elif 'launcher_agent_session_json' in manifest['roots']:
        # Read-only compatibility with Runs created before the participant protocol.
        result['agent_recording'] = session._read(normalized / 'agent/session.json')
    return result
