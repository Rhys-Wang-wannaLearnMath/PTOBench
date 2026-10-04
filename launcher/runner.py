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
from launcher.config import render
from launcher.preflight import preflight
from launcher.prompt import make_prompt


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
        logical = 'launcher_' + name.replace('.', '_')
        manifest = integrity._load(root / 'protection')
        if logical not in manifest['roots']:
            integrity.register_roots(root / 'protection',
                {logical: integrity.ProtectionRoot(root / name, 'records')})
        payload = value.encode() if isinstance(value, str) else session._bytes(value)
        integrity.commit_records(root / 'protection', {logical: payload})


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
    experiment, profile, adapter, info, case = preflight(path)
    optimizer = {**experiment.optimizer, 'command': profile['argv'], 'tool_version': info.version}
    handle = session.create_session(experiment.case, optimizer, experiment.output, experiment.budget)
    root = handle.session_root
    record = session.read_session(root)
    prompt = make_prompt(experiment, record, case)
    environment = {**os.environ, 'PATH': str(common.BENCHMARK_ROOT) + os.pathsep + os.environ.get('PATH', ''),
                   'PTOBENCH_SESSION_ROOT': str(root), 'PTOBENCH_SESSION_ID': record['session_id'],
                   'PTOBENCH_CASE_ID': record['case_id'], 'PTOBENCH_OPTIMIZER_ID': record['optimizer_id'],
                   'PTOBENCH_WORKSPACE': str(handle.initial_workspace)}
    process = None
    logs = None
    active = None
    reason = 'optimizer_exit'
    error = None
    final = None
    code = None
    with signals() as state:
        try:
            save(root, 'experiment.json', experiment.to_dict())
            save(root, 'profile.json', profile)
            save(root, 'prompt.txt', prompt)
            argv, _ = render(profile, handle.initial_workspace, root, prompt)
            provenance = {'backend': info.to_dict(), 'argv': argv, 'cwd': str(handle.initial_workspace),
                          'profile_sha256': profile['sha256']}
            save(root, 'backend.json', provenance)
            logs = Logs(root)
            process = adapter.start(handle.initial_workspace, environment, prompt, profile)
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
                    if (record['usage']['hardware_actions'] >= record['budget']['max_hardware_actions']
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
                    adapter.stop(process)
                    code = process.returncode
            except Exception:
                error = 'infrastructure_error'
            try:
                _stop_action(root, active)
                with session.execution_lock():
                    actions.recover_global_action()
                    actions._recover_unmarked_action(root)
                    if error:
                        session.fail_session(root, reason)
            except Exception:
                error = 'infrastructure_error'
            finally:
                if logs is not None:
                    try:
                        logs.close()
                    except Exception:
                        error = 'infrastructure_error'
        if not error:
            final = finalize_session(root, reason).to_dict()
        try:
            save(root, 'launcher_result.json', {'close_reason': reason, 'optimizer_exit_code': code,
                 'signal': state['received'], 'error_code': error, 'final_result': str(root / 'final_result.json') if final else None})
        except Exception:
            error = 'infrastructure_error'
        exit_code = (130 if state['received'] or final and final['error_code'] == 'interrupted'
                     else 5 if error or final and final['error_code'] == 'infrastructure_error'
                     else 0 if final and final['status'] == 'completed' else 1)
    return {'session_root': str(root), 'final_result_path': str(root / 'final_result.json') if final else None,
            'status': final['status'] if final else 'failed', 'error_code': error or (final or {}).get('error_code'),
            'close_reason': reason, 'optimizer_exit_code': code}, exit_code


def status(root):
    with session._session_lock(root) as normalized:
        record, _ = session._load_locked(normalized)
        manifest = integrity._load(normalized / 'protection')
    result = {key: record[key] for key in ('session_id', 'optimizer_id', 'case_id', 'status',
              'exploration_closed', 'close_reason', 'budget', 'usage', 'active_action')}
    result['remaining_hardware_actions'] = actions._remaining(record)
    result['session_root'] = record['session_root']
    has_final = manifest['roots']['final_result']['entries']['']['kind'] == 'file'
    result['final_result_path'] = record['session_root'] + '/final_result.json' if has_final else None
    return result
