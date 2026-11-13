"""Managed exploration: frozen input, existing Evaluator, events and diagnostics."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import contextlib
import errno
import grp
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time
import uuid

import common
import file_tree
import integrity
import session
import snapshots
import workspace
import action_records

EXIT_CODES = {None: 0, 'action_failed': 1, 'invalid_workspace': 2, 'invalid_request': 2,
              'budget_exceeded': 3, 'session_closed': 3, 'integrity_failed': 4,
              'infrastructure_error': 5, 'interrupted': 130}


@dataclass
class ActionResult:
    status: str
    error_code: str | None
    exit_code: int
    event_seq: int | None = None
    workspace: str | None = None
    correctness: str | None = None
    latency_us: float | None = None
    snapshot_id: str | None = None
    remaining_hardware_actions: int | None = None
    message: str | None = None
    integrity_passed: bool | None = None
    _omit_correctness: bool = False
    failure_diagnostic: dict | None = None
    attempt_id: int | None = None

    def to_dict(self):
        value = asdict(self)
        if value.pop("_omit_correctness"):
            value.pop("correctness")
        if value['failure_diagnostic'] is None:
            value.pop('failure_diagnostic')
        return value


class ActionInterrupted(RuntimeError):
    error_code = 'interrupted'


class ProcessCleanupError(integrity.InfrastructureError):
    """Execution cannot safely proceed to asset repair or another action."""


def _error_result(error, seq=None, path=None, remaining=None):
    code = getattr(error, 'error_code', 'infrastructure_error')
    return ActionResult('REJECTED' if seq is None else 'FAIL', code, EXIT_CODES[code], seq,
                        str(path) if path else None, remaining_hardware_actions=remaining, message=str(error))


def _remaining(record, events):
    maximum = record['budget']['max_hardware_actions']
    if maximum == session.UNLIMITED_HARDWARE_ACTIONS:
        return None
    return maximum - session._evaluate_actions_used(record, events)


def _repair_for_action(root):
    session._require_execution()
    session.read_session(root)  # Record repair stays under the short lock.
    manifest = session._read(Path(root) / 'protection/manifest.json')
    session.require_current_evaluator(manifest)
    groups = {info['group'] for info in manifest['roots'].values()} - {'records'}
    try:
        return integrity.check_and_restore(Path(root) / 'protection', groups)
    except integrity.InfrastructureError:
        session.fail_session(root, 'asset_recovery_failed')
        raise


def _postcheck_action(root, frozen, manifest):
    # Records are restored only under the short Session lock. Include changes
    # found by restoration and status readers, including after check_only.
    report = integrity.check_only(root / 'protection')
    state = session.read_session(root)
    violations = set((state['active_action'] or {}).get('integrity_violations', []))
    try:
        source_intact = workspace.source_manifest(frozen) == manifest
    except workspace.InvalidWorkspaceError:
        source_intact = False
    repair = _repair_for_action(root)
    state = session.read_session(root)
    violations.update((state['active_action'] or {}).get('integrity_violations', []))
    violations.update(set(report.changed_paths) - set(report.blocked_paths))
    violations.update(set(repair.changed_paths) - set(repair.blocked_paths))
    observed = set(report.blocked_paths) | set(repair.blocked_paths)
    repaired = sorted(violations | set(repair.restored_paths))
    return report.passed and source_intact and not violations, repaired, sorted(observed)


def _marker_path():
    return common.BENCHMARK_ROOT / '.ptobench/active_action.json'


def _marker(value):
    file_tree.atomic_bytes(_marker_path(), session._bytes(value))


def _clear_marker():
    with file_tree.open_parent(_marker_path()) as (fd, name):
        file_tree.remove_at(fd, name)
        os.fsync(fd)


def _stop_group(identity, process=None):
    """Terminate only the original owned group; never kill a reused leader PID."""
    if identity is None:
        return
    current = session.process_identity(identity['pid'])
    if current is not None and current != identity:
        raise integrity.InfrastructureError('Evaluator PID was reused; refusing to signal it')
    if identity['pid'] != identity['pgid']:
        raise integrity.InfrastructureError('Evaluator does not own its process group')
    def send(sig):
        try:
            os.killpg(identity['pgid'], sig)
        except ProcessLookupError:
            pass
    if session._group_alive(identity):
        send(signal.SIGTERM)
        deadline = time.monotonic() + 10
        while session._group_alive(identity) and time.monotonic() < deadline:
            if process is not None:
                process.poll()
            time.sleep(0.02)
        if session._group_alive(identity):
            send(signal.SIGKILL)
            deadline = time.monotonic() + 5
            while session._group_alive(identity) and time.monotonic() < deadline:
                if process is not None:
                    process.poll()
                time.sleep(0.02)
    if process is not None:
        process.wait(timeout=5)
    if session._group_alive(identity):
        raise integrity.InfrastructureError('Evaluator group did not stop')


def _stop_recovered_group(root, identity):
    try:
        _stop_group(identity)
    except Exception:
        session.fail_session(root, 'active_process_recovery_failed')
        raise
    try:
        import agent_launch
        agent_launch.cleanup_stale_requests(root)
    except Exception:
        session.fail_session(root, 'agent_request_recovery_failed')
        raise


def _recover_unmarked_action(root):
    # A killed owner can leave the durable reservation before the global spawn
    # marker exists. S2 can finish that known-not-started action once; live or
    # unattributed workers remain blocked instead of being assumed stopped.
    record = session.read_session(root)
    if record['active_action'] is not None:
        session.reconcile_active_action(root)


def recover_global_action():
    """Recover an attributed abandoned execution before any Session executes."""
    session._require_execution()
    path = _marker_path()
    with file_tree.open_directory(path.parent) as fd:
        try:
            os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return
    marker = session._read(path)
    if isinstance(marker, dict) and marker.get('kind') == 'maintenance':
        action_dir = _maintenance_dir(marker.get('action_dir'))
        identity = marker.get('process')
        if (marker.get('schema_version') != 1 or marker.get('state') not in {'starting', 'running'}
                or not isinstance(marker.get('owner'), dict)):
            raise integrity.InfrastructureError('invalid maintenance marker')
        if identity is None:
            raise integrity.InfrastructureError('maintenance spawn has no bound PID; execution blocked')
        if (not isinstance(identity, dict) or set(identity) != {'pid', 'pgid', 'start_ticks'}
                or any(type(value) is not int or value <= 0 for value in identity.values())
                or identity['pid'] != identity['pgid']):
            raise integrity.InfrastructureError('invalid maintenance process identity')
        _stop_group(identity)
        file_tree.atomic_bytes(action_dir / 'process_result.json', session._bytes({
            'exit_code': 130, 'error_code': 'interrupted', 'recovered': True, 'process': identity}))
        _clear_marker()
        return
    if (not isinstance(marker, dict) or marker.get('schema_version') != 1
            or type(marker.get('event_seq')) is not int or not isinstance(marker.get('session_root'), str)):
        raise integrity.InfrastructureError('invalid global active-action marker')
    root = session._path(marker['session_root'], code='invalid_request')
    record = session.read_session(root)
    active = record['active_action']
    if active is None:
        with session._session_lock(root):
            _, events = session._load_locked(root)
        if not any(event['seq'] == marker['event_seq'] for event in events):
            raise integrity.InfrastructureError('global marker has no matching active/completed action')
        _stop_recovered_group(root, marker.get('process'))
        _clear_marker()
        return
    if active['event_seq'] != marker['event_seq'] or active['owner'] != marker.get('owner'):
        raise integrity.InfrastructureError('global marker does not match the protected action')
    identity = marker.get('process')
    if identity is None and marker.get('state') == 'starting':
        # The owner writes the marker before Popen()/PID binding.  If that
        # owner dies in this narrow window, the old implementation left the
        # marker and the reserved action behind forever, blocking every later
        # Session.  Recover only when the marker's owner is no longer the
        # same live process; a live or ambiguous owner remains fail-closed.
        owner = marker.get('owner')
        if (not isinstance(owner, dict) or set(owner) != {'pid', 'pgid', 'start_ticks'}
                or any(type(value) is not int or value <= 0 for value in owner.values())):
            session.fail_session(root, 'unattributed_spawn')
            raise integrity.InfrastructureError('spawn interrupted before PID binding; execution blocked')
        current_owner = session.process_identity(owner['pid'])
        if current_owner == owner:
            session.fail_session(root, 'unattributed_spawn')
            raise integrity.InfrastructureError('spawn interrupted before PID binding; execution blocked')
        # Reconcile_active_action deliberately treats process_state=starting
        # as ambiguous.  Here the durable global marker and matching Session
        # record prove that no child PID was ever bound, and the owner birth
        # identity is gone, so downgrade it to the safe pre-spawn state before
        # using the normal one-time interrupted-action reconciliation.
        with session._session_lock(root):
            record, _ = session._load_locked(root)
            pending = record['active_action']
            if (pending is None or pending['event_seq'] != marker['event_seq']
                    or pending['owner'] != owner or pending['process'] is not None):
                raise integrity.InfrastructureError('unattributed spawn does not match the protected action')
            pending['process_state'] = 'not_started'
            session._commit(root, record)
        session.reconcile_active_action(root)
        _clear_marker()
        return
    if identity is not None and active['process'] not in (None, identity):
        raise integrity.InfrastructureError('global and Session process identities disagree')
    _stop_recovered_group(root, identity)
    action_dir = root / 'final' if active['kind'] == 'finish' else root / 'staging' / f"action_{active['event_seq']:06d}"
    frozen = action_dir / 'workspace'
    try:
        manifest = session._read(frozen.parent / 'source_manifest.json')
    except integrity.InfrastructureError:
        # Recover the protected manifest before loading it, retaining witnesses.
        session.read_session(root)
        manifest = session._read(frozen.parent / 'source_manifest.json')
    intact, repaired, observed_mode_changes = _postcheck_action(root, frozen, manifest)
    session.close_exploration(root, 'interrupted')
    session.complete_action(root, active['event_seq'], exit_code=130, error_code='interrupted',
                            integrity_passed=intact, repaired_paths=repaired,
                            action_started=marker.get('process') is not None,
                            observed_mode_changes=observed_mode_changes,
                            artifact_index=action_records.archive(root, active['event_seq'], action_dir))
    _clear_marker()


@contextlib.contextmanager
def _signals():
    previous = {}
    state = {'raise': True, 'received': None, 'active': True}
    def interrupted(signum, frame):
        if state['active']:
            state['received'] = state['received'] or signum
            if state['raise']:
                state['raise'] = False
                raise ActionInterrupted(f'received signal {signum}')
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupted)
        yield state
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _run_process(argv, environment, action_dir, root, seq, signal_state):
    """Drain both pipes to disk, with nonblocking terminal forwarding."""
    session._require_execution()
    selector = selectors.DefaultSelector()
    process = None
    identity = None
    process_started_at = session._now()
    log_files = []
    outputs = []
    marker = {'schema_version': 1, 'session_root': str(root), 'event_seq': seq,
              'owner': session.process_identity(), 'process': None, 'state': 'starting'}
    if root is None:
        marker = {'schema_version': 1, 'kind': 'maintenance', 'action_dir': str(action_dir),
                  'owner': session.process_identity(), 'process': None, 'state': 'starting'}
    _marker(marker)
    if root is not None:
        session.mark_action_starting(root, seq)
    gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
    environment = {**environment, 'PTOBENCH_ACTION_START_FD': str(gate_read)}
    try:
        process = subprocess.Popen(argv, shell=False, env=environment, cwd=action_dir / 'workspace',
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                                   pass_fds=(gate_read,))
        identity = session.process_identity(process.pid)
        if identity is None:
            # A very short child may already be a zombie; /proc still provides
            # its birth identity so it can be attributed without assuming success.
            data = Path(f'/proc/{process.pid}/stat').read_text()
            fields = data[data.rfind(')') + 2:].split()
            identity = {'pid': process.pid, 'start_ticks': int(fields[19]), 'pgid': int(fields[2])}
        marker.update(process=identity, state='running')
        _marker(marker)
        # Fast exited children do not need a live-PID bind API. Commit their
        # pinned birth identity before reaping, under the short record lock.
        if root is not None:
            with session._session_lock(root):
                record, _ = session._load_locked(root)
                record['active_action']['process'] = identity
                record['active_action']['process_state'] = 'running'
                session._commit(root, record)
        else:
            file_tree.atomic_bytes(action_dir / 'process_identity.json', session._bytes(identity))
        for pipe, stream, name in ((process.stdout, sys.stdout, 'stdout.log'), (process.stderr, sys.stderr, 'stderr.log')):
            with file_tree.open_parent(action_dir / name) as (parent, leaf):
                fd = os.open(leaf, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                             0o600, dir_fd=parent)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    os.close(fd)
                    raise integrity.InfrastructureError('action log is not a regular file')
                log = os.fdopen(fd, 'w+b', buffering=0)
            log_files.append(log)
            selector.register(pipe, selectors.EVENT_READ, log)
            try:
                fd = stream.fileno()
                blocking = os.get_blocking(fd)
                os.set_blocking(fd, False)
                outputs.append([fd, blocking, log, 0, False])
            except (AttributeError, OSError, ValueError):
                outputs.append([None, None, log, 0, True])
        # The existing Evaluator waits before resolving/importing a Case. Its
        # PID/records/logs are durable before it can edit any protected record.
        os.write(gate_write, b'G')
        os.close(gate_write)
        gate_write = None
        leader_stopped = False
        while selector.get_map():
            for key, _ in selector.select(timeout=0.05):
                data = os.read(key.fd, 65536)
                if data:
                    key.data.seek(0, os.SEEK_END)
                    key.data.write(data)
                else:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            _forward(outputs)
            if process.poll() is not None and not leader_stopped:
                _stop_group(identity, process)
                leader_stopped = True
        code = process.wait()
        _stop_group(identity, process)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and _forward(outputs):
            time.sleep(0.005)
        return code
    finally:
        # Repeated signals cannot interrupt owned-group cleanup or postchecks.
        signal_state['raise'] = False
        os.close(gate_read)
        if gate_write is not None:
            os.close(gate_write)
        # No postcheck may run while a child or descendant is still executing.
        cleanup_error = None
        if process is not None:
            try:
                if identity is None:
                    process.kill()
                    process.wait(timeout=5)
                    raise ProcessCleanupError('Evaluator group identity could not be established')
                _stop_group(identity, process)
            except Exception as error:
                cleanup_error = ProcessCleanupError(f'Evaluator cleanup failed: {error}')
            if cleanup_error is None:
                for pipe, log in zip((process.stdout, process.stderr), log_files):
                    if pipe is not None and not pipe.closed:
                        os.set_blocking(pipe.fileno(), False)
                        while True:
                            try:
                                data = os.read(pipe.fileno(), 65536)
                            except BlockingIOError:
                                break
                            if not data:
                                break
                            log.seek(0, os.SEEK_END)
                            log.write(data)
                _forward(outputs)
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        selector.close()
        for fd, blocking, _, _, _ in outputs:
            if fd is not None:
                os.set_blocking(fd, blocking)
        for log in log_files:
            log.close()
        file_tree.atomic_bytes(action_dir / 'process_execution.json', session._bytes({
            'schema_version': 1, 'started_at': process_started_at, 'ended_at': session._now(),
            'process': identity, 'spawned': process is not None,
            'exit_code': process.returncode if process is not None else None,
            'received_signal': signal_state.get('received'),
            'cleanup_error': str(cleanup_error) if cleanup_error else None,
        }))
        if cleanup_error is not None:
            raise cleanup_error


def _maintenance_dir(value):
    if not isinstance(value, (str, Path)):
        raise integrity.InfrastructureError('invalid maintenance action directory')
    path = session._path(value, code='invalid_request')
    runs = file_tree.absolute_path(common.BENCHMARK_ROOT / 'runs')
    if path == runs or not path.is_relative_to(runs):
        raise integrity.InfrastructureError('maintenance evidence must be below benchmark runs')
    session._path(path / 'workspace', code='invalid_request')
    return path


def run_maintenance_process(argv, environment, action_dir):
    """Internal control execution; caller holds the lock, no Session budget.

    Shares the Evaluator gate, log draining, birth identity and group cleanup
    with exploration/finish. An abandoned worker is recovered under that same
    lock before the next action. This is not an Agent CLI or score publisher.
    """
    session._require_execution()
    action_dir = _maintenance_dir(action_dir)
    recover_global_action()
    if os.path.lexists(action_dir / 'command.json'):
        raise integrity.InfrastructureError('maintenance evidence directory already used')
    file_tree.atomic_bytes(action_dir / 'command.json', session._bytes({'argv': argv,
                          'cwd': str(action_dir / 'workspace')}))
    safe = True
    with _signals() as state:
        try:
            code = _run_process(argv, environment, action_dir, None, None, state)
            file_tree.atomic_bytes(action_dir / 'process_result.json', session._bytes({
                'exit_code': code, 'error_code': None, 'recovered': False}))
            return code
        except ProcessCleanupError:
            safe = False
            raise
        except BaseException as error:
            file_tree.atomic_bytes(action_dir / 'process_result.json', session._bytes({
                'exit_code': 130 if state['received'] else 5,
                'error_code': getattr(error, 'error_code', 'infrastructure_error'), 'recovered': False}))
            raise
        finally:
            if safe:
                _clear_marker()


def _forward(outputs):
    pending = False
    for output in outputs:
        fd, _, log, offset, dead = output
        if dead:
            continue
        log.seek(0, os.SEEK_END)
        end = log.tell()
        if offset >= end:
            continue
        pending = True
        log.seek(offset)
        data = log.read(min(65536, end - offset))
        try:
            output[3] += os.write(fd, data)
        except BlockingIOError:
            pass
        except OSError as error:
            if error.errno in (errno.EPIPE, errno.EBADF):
                output[4] = True
            else:
                raise
    return pending


def _positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _attempt_metadata(action_dir, kind, error_code, observation, raw, artifact_index):
    """Flatten the trusted Evaluator result into fields useful for offline analysis."""
    execution_path = action_dir / 'execution.json'
    try:
        execution = session._read(execution_path)
    except (integrity.InfrastructureError, OSError):
        execution = None
    worker_diagnostics = action_dir / 'worker_diagnostics'
    worker_records = []
    worker_executions = []
    if worker_diagnostics.is_dir():
        for path in sorted(worker_diagnostics.glob('*/worker_process.json')):
            try:
                worker_records.append(session._read(path))
            except (integrity.InfrastructureError, OSError):
                continue
        for path in sorted(worker_diagnostics.glob('*/worker_execution.json')):
            try:
                worker_executions.append(session._read(path))
            except (integrity.InfrastructureError, OSError):
                continue
    stage_sources = worker_executions or ([execution] if execution else [])
    stages = [event for source in stage_sources for event in source.get('events', [])
              if event.get('stage') and event.get('event_type') == 'stage']

    def stage_status(name):
        completed = False
        for source in stage_sources:
            events = [event for event in source.get('events', [])
                      if event.get('stage') == name and event.get('event_type') == 'stage']
            if not events:
                continue
            status = events[-1].get('status')
            if status == 'failed':
                return 'fail'
            if status == 'started':
                return 'error'
            completed |= status == 'passed'
        if completed:
            return 'pass'
        order = ('compile', 'runtime', 'validate', 'benchmark')
        if name == 'benchmark' and kind == 'profile':
            return 'not_run'
        if name == 'validate' and skipped:
            return 'not_run'
        if name in order:
            earlier = order[:order.index(name)]
            if any(stage_status(stage) in {'fail', 'error'} for stage in earlier):
                return 'not_run'
        return None

    skipped = any(source.get('skip_correctness') is True for source in stage_sources)
    compile_status = stage_status('compile') or 'unavailable'
    runtime_status = stage_status('runtime') or 'unavailable'
    benchmark_status = stage_status('benchmark') or 'unavailable'
    validation = stage_status('validate')
    try:
        evaluator_result = raw or session._read(action_dir / 'result.json')
    except (integrity.InfrastructureError, OSError):
        evaluator_result = {}
    result_runs = evaluator_result.get('runs', []) if isinstance(evaluator_result, dict) else []
    result_passed = (evaluator_result.get('verdict') == 'PASS' and result_runs
                     and all(run.get('passed') is True for run in result_runs))
    if result_passed:
        compile_status = 'pass' if compile_status == 'unavailable' else compile_status
        runtime_status = 'pass' if runtime_status == 'unavailable' else runtime_status
    if skipped:
        correctness_status = 'not_run'
    elif validation:
        correctness_status = validation
    elif compile_status in {'fail', 'error'} or runtime_status in {'fail', 'error'}:
        correctness_status = 'not_run'
    elif result_passed:
        correctness_status = 'pass'
    elif validation == 'fail' or evaluator_result.get('verdict') == 'FAIL':
        correctness_status = 'fail'
    else:
        correctness_status = 'unavailable'

    failed_stage = next((event['stage'] for event in reversed(stages)
                         if event.get('status') == 'failed'), None)
    if failed_stage is None:
        failed_stage = next((name for name in ('compile', 'runtime', 'validate', 'benchmark')
                             if stage_status(name) == 'error'), None)
    failure_stage = {'validate': 'correctness', 'compute golden': 'oracle',
                     'generate inputs': 'input_setup'}.get(failed_stage, failed_stage)
    worker_missing_result = any(not item.get('summary_present') for item in worker_records)
    if failure_stage is None and worker_missing_result:
        failure_stage = 'worker'
    if failure_stage is None and error_code:
        failure_stage = 'unavailable'
    runs = raw.get('runs', []) if isinstance(raw, dict) else []
    measured = (kind in {'evaluate', 'finish'} and correctness_status == 'pass'
                and raw and raw.get('verdict') == 'PASS'
                and all(run.get('passed') is True for run in runs))
    performance = {}
    flattened_samples = []
    device_samples = {}
    measurements = {}
    device_time = None
    latency = speedup = None
    if measured:
        for run in runs:
            fixture = run['fixture_id']
            samples = list(run.get('samples_effective_us', []))
            performance[fixture] = samples
            flattened_samples.extend(samples)
            measurements[fixture] = run.get('measurement_config')
            samples = run.get('samples_device_us', [])
            device_samples[fixture] = list(samples)
        aggregate = common.fixture_aggregate(runs)
        latency, speedup = aggregate['latency_us'], aggregate['speedup']
        device_values = [value for samples in device_samples.values() for value in samples
                         if _positive(value)]
        device_time = sum(device_values) if device_values else None

    artifact_root = str(Path(artifact_index).parent) if artifact_index else None
    worker_log_paths = []
    if artifact_root and worker_diagnostics.is_dir():
        worker_log_paths = [str(Path(artifact_root) / path.relative_to(action_dir))
                            for path in sorted(worker_diagnostics.glob('*/*.log'))]
    runtime_dispatch_entries = None
    if execution:
        runtime_dispatch_entries = sum(event.get('stage') == 'runtime_dispatch_entered'
                                        for event in execution.get('events', []))
    if kind == 'profile':
        benchmark_status = 'not_run'
    elif measured:
        benchmark_status = 'pass'
    elif benchmark_status == 'pass' and not flattened_samples:
        benchmark_status = 'unavailable'
    elif benchmark_status == 'not_run' and result_passed:
        benchmark_status = 'unavailable'
    return {
        'compile_status': compile_status,
        'runtime_status': runtime_status,
        'correctness_status': correctness_status,
        'failure_stage': failure_stage,
        'worker_status': ('missing_result' if worker_missing_result else
                          'result_written' if worker_records else 'unavailable'),
        'worker_exit_codes': [item.get('exit_code') for item in worker_records],
        'worker_log_paths': worker_log_paths,
        'bench_status': benchmark_status,
        'diagnostic_path': artifact_root,
        'latency_samples': flattened_samples,
        'latency_samples_by_fixture': performance,
        'latency_us': latency,
        'speedup': speedup,
        'measurement_config': measurements,
        'device_latency_samples_us': device_samples,
        'bench_sampled_device_time_us': device_time,
        'runtime_dispatch_entries': runtime_dispatch_entries,
        'bench_sampled_rounds': sum(len(samples) for samples in device_samples.values()) or None,
        'profile_artifacts_path': artifact_root if kind == 'profile' else None,
        'effective_compile_options': raw.get('effective_tuning') if isinstance(raw, dict) else None,
        'agent_session_path': 'agent/session.json',
        'agent_dispatch_id': os.environ.get('PTOBENCH_AGENT_DISPATCH_ID'),
        'agent_parent_tool_call_id': os.environ.get('PTOBENCH_AGENT_PARENT_TOOL_CALL_ID'),
        'agent_parent_tool_call_id_status': ('available' if os.environ.get('PTOBENCH_AGENT_PARENT_TOOL_CALL_ID')
                                             else 'unavailable'),
    }


def validate_result(path, record, frozen, kind, internal_mode, exit_code):
    value = session._read(path)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise session.SessionError('nonfinite or malformed result JSON', 'action_failed') from error
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int or value.get('schema_version') != 1
            or value.get('case_id') != record['case_id'] or value.get('candidate') != str(frozen)
            or value.get('phase') != ('all' if kind == 'evaluate' else 'correctness')
            or value.get('profiling_mode') != internal_mode
            or value.get('official_score') is not (kind == 'evaluate')
            or value.get('diagnostic_only') is not (kind != 'evaluate')
            or value.get('verdict') not in {'PASS', 'FAIL'}
            or not isinstance(value.get('runs'), list) or not value['runs']):
        raise session.SessionError('invalid Evaluator result identity/schema', 'action_failed')
    declared = common.declared_fixture_ids(session._read(Path(record['case_dir']) / 'case.json'))
    runs = value['runs']
    if (any(not isinstance(run, dict) or type(run.get('passed')) is not bool for run in runs)
            or len(runs) != len(declared) or {run.get('fixture_id') for run in runs} != set(declared)):
        raise session.SessionError('invalid or incomplete fixture result set', 'action_failed')
    passed = value['verdict'] == 'PASS' and all(run['passed'] for run in runs) and exit_code == 0
    if not passed:
        raise session.SessionError('Evaluator/correctness failed', 'action_failed')
    diagnostic_only = kind == 'profile' and record['optimizer']['backend'] != 'maintenance'
    for run in runs:
        expected = common.fixture_input_identity(Path(record['sealed_dir'])/'fixtures'/run['fixture_id'])
        if run.get('input_identity_sha256') != expected:
            raise session.SessionError('fixture input identity does not match sealed inputs', 'action_failed')
        if kind == 'evaluate':
            if not _positive(run.get('median_effective_us')) or not _positive(run.get('speedup')):
                raise session.SessionError('clean result requires finite positive latency and speedup', 'action_failed')
            samples = run.get('samples_effective_us')
            raw_samples = run.get('raw_samples_effective_us')
            config = run.get('measurement_config')
            if (not isinstance(samples, list) or not samples or not all(_positive(sample) for sample in samples)
                    or not isinstance(raw_samples, list) or raw_samples != samples
                    or not isinstance(config, dict) or config.get('unit') != 'us' or config.get('metric') != 'effective'):
                raise session.SessionError('clean result must retain original timing samples and settings', 'action_failed')
    if kind == 'evaluate':
        aggregate = common.fixture_aggregate(runs)
        latency, metric = aggregate['latency_us'], aggregate['metric_name']
    else:
        # Maintenance diagnostic selection is meaningful only for one fixture;
        # diagnostic timing is never pooled into a multi-input performance score.
        latency = runs[0].get('median_effective_us') if len(runs) == 1 else None
        metric = 'median_effective_us'
    latency = float(latency) if not diagnostic_only and _positive(latency) else None
    observation = {'kind': kind, 'mode': None if kind == 'evaluate' else next(k for k, v in session.PUBLIC_MODES.items() if v == internal_mode),
                   'latency_us': latency,
                   'metric_name': metric if latency is not None else None,
                   'unit': 'us' if latency is not None else None}
    if not diagnostic_only:
        observation['correctness'] = 'PASS'
    if len(runs) > 1:
        observation.update(fixture_count=len(runs),
            fixture_metrics=[{'fixture_id':r['fixture_id'], 'latency_us':r.get('median_effective_us')
                              if kind == 'evaluate' else None} for r in runs], end_to_end=False)
    return value, observation


def _publish_diagnostics(original, frozen, internal_mode, action_result, raw_result, *, root, artifact_index):
    """Replace complete output sets, removing links themselves without following."""
    share_gid = grp.getgrnam('ptobench-share').gr_gid
    saved = action_records.verified_archive(root, artifact_index, event_seq=action_result.event_seq)
    source = saved / '.benchmark_outputs' / internal_mode
    parent = original / '.benchmark_outputs'
    with file_tree.open_parent(parent) as (fd, name):
        try:
            mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None and not stat.S_ISDIR(mode):
            file_tree.remove_at(fd, name)
    with file_tree.open_directory(parent, create=True) as fd:
        os.fchown(fd, -1, share_gid)
        os.fchmod(fd, 0o750)
        previous = parent / internal_mode
        previous_entries = file_tree.scan_tree(previous, excluded_names=frozenset())
        if previous_entries.get('', {}).get('kind') == 'directory':
            receipt = session._read(previous / 'capture_receipt.json')
            if (not isinstance(receipt, dict) or receipt.get('mode') != internal_mode
                    or not isinstance(receipt.get('artifact_index'), str)
                    or type(receipt.get('event_seq')) is not int):
                raise integrity.InfrastructureError('previous capture mode mismatch; originals retained')
            old = action_records.verified_archive(root, receipt['artifact_index'], event_seq=receipt['event_seq'])
            old_entries = file_tree.scan_tree(old / '.benchmark_outputs' / internal_mode, excluded_names=frozenset())
            # Ignore only the public publication metadata, never trace content.
            metadata = {'latest_result.json', 'latest_session_result.json', 'capture_receipt.json'}
            def content(entries):
                return {p: {k: v for k, v in e.items() if k != 'mode'}
                        for p, e in entries.items() if p not in metadata}
            old_content = content(old_entries)
            if {p: e for p, e in content(previous_entries).items() if p in old_content} != old_content:
                raise integrity.InfrastructureError('previous capture differs from saved originals; retained')
        temporary = parent / f'.{internal_mode}.publish_{uuid.uuid4().hex}'
        with file_tree.open_directory(temporary, create=True):
            pass
        try:
            try:
                source_context = file_tree.open_parent(source)
                with source_context as (source_parent, source_name):
                    try:
                        os.stat(source_name, dir_fd=source_parent, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        workspace.copy_tree(source, temporary)
            except FileNotFoundError:
                pass
            if raw_result is None:
                try:
                    with file_tree.open_file(frozen.parent / 'result.json') as raw_fd, os.fdopen(os.dup(raw_fd), 'rb') as reader:
                        file_tree.atomic_bytes(temporary / 'latest_result.json', reader.read(), mode=0o640)
                except (OSError, file_tree.UnsafeTreeError):
                    pass
            file_tree.atomic_bytes(temporary / 'latest_session_result.json',
                                   session._bytes(action_result.to_dict()), mode=0o640)
            if raw_result is not None:
                file_tree.atomic_bytes(temporary / 'latest_result.json',
                                       session._bytes(raw_result), mode=0o640)
            file_tree.atomic_bytes(temporary / 'capture_receipt.json', session._bytes({
                'schema_version': 1, 'event_seq': action_result.event_seq,
                'mode': internal_mode, 'artifact_index': artifact_index}), mode=0o640)
            # The public copy is Broker-owned feedback; preserve artifact
            # content while normalizing every published entry to read-only DAC.
            workspace._readonly(temporary)
            file_tree.remove_at(fd, internal_mode)
            os.rename(temporary.name, internal_mode, src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
        finally:
            file_tree.remove_at(fd, temporary.name)


def reject_request(session_root, action, error, *, path=None, mode=None):
    try:
        with session.execution_lock():
            recover_global_action()
            with session._session_lock(session._path(session_root, code='invalid_request')) as root:
                record, _ = session._load_locked(root)
            _recover_unmarked_action(record['session_root'])
            code = getattr(error, 'error_code', 'invalid_request')
            event = session.record_nonhardware_event(session_root, action, mode=mode,
                     error_code=code, exit_code=EXIT_CODES[code], result={'message': str(error)})
            with session._session_lock(Path(record['session_root'])) as root:
                current, events = session._load_locked(root)
            result = _error_result(error, event['seq'], path, _remaining(current, events))
            result.status = 'REJECTED'
            result.attempt_id = event.get('attempt_id')
            return result
    except (session.SessionError, integrity.InfrastructureError, OSError, ValueError) as failure:
        return _error_result(failure)


def run_evaluate(session_root, workspace_path):
    return _run_action(session_root, workspace_path, 'evaluate', None)


def run_profile(session_root, workspace_path, public_mode):
    result = _run_action(session_root, workspace_path, 'profile', public_mode)
    return result


def _run_action(session_root, workspace_path, kind, mode):
    try:
        with session.execution_lock(), _signals() as signal_state:
            recover_global_action()
            root = session._path(session_root, code='invalid_request')
            seq = None
            accepted = False
            frozen = None
            repaired = []
            observed_mode_changes = []
            record = None
            try:
                repairs = _repair_for_action(root)
                repaired += repairs.restored_paths
                observed_mode_changes += repairs.blocked_paths
                _recover_unmarked_action(root)
                with session._session_lock(root) as normalized:
                    record, events = session._load_locked(normalized)
                session._open_exploration(record)
                maximum = record['budget']['max_hardware_actions']
                if (kind == 'evaluate' and maximum != session.UNLIMITED_HARDWARE_ACTIONS
                        and session._evaluate_actions_used(record, events) >= maximum):
                    raise session.SessionError('Evaluate allowance exhausted', 'budget_exceeded')
                worker_id, original, _ = session.resolve_workspace(root, workspace_path)
                if kind == 'profile' and (mode not in session.PUBLIC_MODES or mode not in record['profiling_modes']):
                    raise session.SessionError('profile is not enabled')
                seq = record['next_event_seq']
                action_dir = root / 'staging' / f'action_{seq:06d}'
                with file_tree.open_directory(action_dir, create=True):
                    pass
                frozen = action_dir / 'workspace'
                manifest = workspace.copy_candidate_tree(original, frozen / 'candidate')
                case = session._read(Path(record['case_dir']) / 'case.json')
                try:
                    compile_config = common.read_tuning(frozen, case)
                except ValueError as error:
                    raise session.SessionError(str(error)) from error
                workspace.write_context(Path(record['case_dir']), Path(record['sealed_dir']), case, frozen,
                                        parent_workspace=original, **session.context_options(record))
                integrity.register_roots(root / 'protection', {
                    f'action_context_{seq}': integrity.ProtectionRoot(frozen / '.benchmark_context', 'action_context'),
                    f'action_source_{seq}': integrity.ProtectionRoot(action_dir / 'source_manifest.json', 'records'),
                    f'action_result_{seq}': integrity.ProtectionRoot(action_dir / 'authority_result.json', 'records'),
                })
                with session._session_lock(root):
                    session._load_locked(root)
                    integrity.commit_records(root / 'protection', {f'action_source_{seq}': session._bytes(manifest)})
                session.reserve_action(root, seq, kind, workspace_path=original, mode=mode,
                                       source_manifest=manifest,
                                       observed_mode_changes=observed_mode_changes,
                                       compile_config=compile_config)
                accepted = True
            except (session.SessionError, workspace.InvalidWorkspaceError) as error:
                event = session.record_nonhardware_event(root, kind, mode=mode,
                         error_code=error.error_code, exit_code=EXIT_CODES[error.error_code], repaired_paths=repaired,
                         observed_mode_changes=observed_mode_changes, result={'message': str(error)})
                with session._session_lock(root) as normalized:
                    current, current_events = session._load_locked(normalized)
                result = _error_result(error, event['seq'], workspace_path,
                                       _remaining(current, current_events))
                result.status = 'REJECTED'
                result.attempt_id = event.get('attempt_id')
                return result
            internal_mode = 'clean' if kind == 'evaluate' else session.PUBLIC_MODES[mode]
            argv = [sys.executable, str(common.BENCHMARK_ROOT / 'evaluator/evaluate.py'),
                    '--case', record['case_id'], '--candidate', str(frozen), '--phase',
                    'all' if kind == 'evaluate' else 'correctness', '--profile-mode', internal_mode,
                    '--result', str(action_dir / 'result.json')]
            if kind == 'evaluate':
                argv.extend(['--measurement-mode', 'exploration'])
            elif record['optimizer']['backend'] != 'maintenance':
                argv.append('--skip-correctness')
            environment = os.environ.copy()
            environment.pop('PYPTO_BENCH_ENV', None)
            environment.pop('PYPTO_BENCH_PROFILE', None)
            environment.pop('PTOBENCH_INCORE_KERNEL', None)
            environment.pop('PTOBENCH_INCORE_INPUTS', None)
            raw = None
            observation = None
            error_code = None
            message = None
            evaluator_exit = 1
            started = False
            intact = False
            unsafe_stop = False
            try:
                started = True
                evaluator_exit = _run_process(argv, environment, action_dir, root, seq, signal_state)
            except ProcessCleanupError:
                unsafe_stop = True
                raise
            except (ActionInterrupted, KeyboardInterrupt) as error:
                error_code, message, evaluator_exit = 'interrupted', str(error), 130
                session.close_exploration(root, 'interrupted')
            except Exception as error:
                error_code, message = 'infrastructure_error', str(error)
                try:
                    started = session._read(_marker_path()).get('process') is not None
                except integrity.InfrastructureError:
                    pass
            finally:
                # The execution helper has stopped every owned process before
                # returning/raising. Always inspect before repairing evidence.
                if not unsafe_stop:
                    intact, changes, observed_changes = _postcheck_action(root, frozen, manifest)
                    repaired += changes
                    observed_mode_changes += observed_changes
                    if not intact and error_code != 'interrupted':
                        error_code = 'integrity_failed'
                        message = 'protected content or frozen source changed during execution'
            if signal_state['received']:
                error_code, message, evaluator_exit = 'interrupted', f"received signal {signal_state['received']}", 130
                session.close_exploration(root, 'interrupted')
            if error_code is None:
                try:
                    raw, observation = validate_result(action_dir / 'result.json', record, frozen, kind, internal_mode,
                                                       evaluator_exit)
                except (session.SessionError, integrity.InfrastructureError, OSError, ValueError) as error:
                    error_code, message = 'action_failed', str(error)
            snapshot_id = None
            # Publication is an atomic completion decision. Defer signals until
            # all authority, Snapshot and event commits finish consistently.
            signal_state['active'] = False
            try:
                with file_tree.open_file(action_dir / 'result.json') as fd, os.fdopen(os.dup(fd), 'rb') as stream:
                    original_result = stream.read()
            except FileNotFoundError:
                original_result = None
            if original_result is not None:
                with session._session_lock(root):
                    session._load_locked(root)
                    integrity.commit_records(root / 'protection', {f'action_result_{seq}': original_result})
            artifact_index = action_records.archive(root, seq, action_dir)
            attempt_metadata = _attempt_metadata(action_dir, kind, error_code, observation, raw, artifact_index)
            if error_code is None and observation.get('correctness') == 'PASS':
                snapshot_id = snapshots.create_snapshot(root, frozen, {
                    'seq': seq, 'workspace': str(original), 'content_sha256': manifest['content_sha256'],
                    'exit_code': evaluator_exit, 'correctness': 'PASS', 'integrity_passed': intact,
                    'observation': observation,
                })
            event = session.complete_action(root, seq, exit_code=evaluator_exit, error_code=error_code,
                    integrity_passed=intact, result=raw, repaired_paths=repaired, snapshot_id=snapshot_id,
                    action_started=started, observation=observation,
                    observed_mode_changes=observed_mode_changes, artifact_index=artifact_index,
                    attempt_metadata=attempt_metadata)
            _clear_marker()
            with session._session_lock(root) as normalized:
                current, current_events = session._load_locked(normalized)
            result = ActionResult('PASS' if error_code is None else 'FAIL', error_code, EXIT_CODES[error_code], seq,
                                  str(original), observation.get('correctness') if observation else None,
                                  observation['latency_us'] if observation else None, snapshot_id,
                                  _remaining(current, current_events), message)
            result._omit_correctness = kind == 'profile' and record['optimizer']['backend'] != 'maintenance'
            result.integrity_passed = intact
            result.attempt_id = event.get('attempt_id')
            if error_code == 'infrastructure_error':
                session.fail_session(root, 'action_infrastructure_failure')
            try:
                _publish_diagnostics(original, frozen, internal_mode, result, raw, root=root, artifact_index=artifact_index)
            except (OSError, ValueError, integrity.InfrastructureError) as error:
                # Diagnostic publication cannot change already committed evidence.
                result.message = f'diagnostics unavailable: {error}'
            return result
    except (Exception, KeyboardInterrupt) as error:
        # A broken authoritative commit or repair is never published as success.
        try:
            with session.execution_lock():
                session.fail_session(session_root, 'action_infrastructure_failure')
        except Exception:
            pass
        result = _error_result(error, seq if locals().get('accepted') else None)
        if locals().get('accepted'):
            result.status = 'FAIL'
        return result


def fork_workspace(session_root, parent_path):
    try:
        with session.execution_lock():
            recover_global_action()
            with session._session_lock(session._path(session_root, code='invalid_request')) as normalized:
                record, previous_events = session._load_locked(normalized)
            repairs = _repair_for_action(record['session_root'])
            _recover_unmarked_action(record['session_root'])
            try:
                parent_id, parent, _ = session.resolve_workspace(session_root, parent_path)
                parent_attempt_id = next((event.get('attempt_id') for event in reversed(previous_events)
                    if event.get('workspace') == parent_id and event.get('source_snapshot') is not None
                    and event.get('attempt_id') is not None), None)
                name = session.fork_workspace_locked(session_root, parent_path)
                child = Path(record['session_root']) / 'workspaces' / name
                event = session.record_nonhardware_event(session_root, 'fork', workspace_id=name,
                                                        repaired_paths=repairs.restored_paths,
                                                        observed_mode_changes=repairs.blocked_paths,
                                                        result={'workspace': str(child),
                                                                'child_workspace_id': name,
                                                                'parent_workspace_id': parent_id,
                                                                'parent_attempt_id': parent_attempt_id,
                                                                'source_snapshot': str((child / '.benchmark_context' / 'starting_point').relative_to(record['session_root']))})
                with session._session_lock(Path(record['session_root'])) as normalized:
                    current, current_events = session._load_locked(normalized)
                return ActionResult('PASS', None, 0, event['seq'], str(child),
                                    remaining_hardware_actions=_remaining(current, current_events))
            except (session.SessionError, workspace.InvalidWorkspaceError) as error:
                event = session.record_nonhardware_event(session_root, 'fork', error_code=error.error_code,
                                                        exit_code=EXIT_CODES[error.error_code], result={'message': str(error)})
                with session._session_lock(Path(record['session_root'])) as normalized:
                    current, current_events = session._load_locked(normalized)
                result = _error_result(error, event['seq'], parent_path,
                                       _remaining(current, current_events))
                result.status = 'REJECTED'
                return result
    except Exception as error:
        if not isinstance(error, (session.SessionError, workspace.InvalidWorkspaceError)):
            try:
                with session.execution_lock():
                    session.fail_session(session_root, 'fork_infrastructure_failure')
            except Exception:
                pass
        return _error_result(error)
