"""S4 internal finalization, using the caller-held S3 execution machinery."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys

import common
import file_tree
import integrity
import session
import session_actions as actions
import snapshots
import workspace


@dataclass
class FinalResult:
    schema_version: int
    session_id: str | None
    optimizer_id: str | None
    case_id: str | None
    status: str
    close_reason: str | None
    selection_policy: str
    selection_reason: str | None
    selected_snapshot: str | None
    selected_event_seq: int | None
    selected_content_sha256: str | None
    selected_observation: dict | None
    best_observed_source: str | None
    best_observed_mode: str | None
    best_observed_latency_us: float | None
    best_observed_metric_name: str | None
    best_observed_unit: str | None
    final_verified_correctness: str | None
    final_verified_latency_us: float | None
    final_verified_speedup: float | None
    final_integrity_passed: bool
    hardware_actions_used: int
    finish_actions_used: int
    finish_event_seq: int | None
    error_code: str | None
    completed_at: str | None = None
    message: str | None = None

    def to_dict(self):
        return asdict(self)


def _result(record, selected=None, *, error_code=None, raw=None, intact=False, seq=None, message=None):
    observation = selected.observation if selected else {}
    passed = error_code is None and raw is not None and intact
    run = raw['runs'][0] if passed else {}
    usage = record.get('usage', {'hardware_actions': 0, 'finish_actions': 0})
    return FinalResult(1, record.get('session_id'), record.get('optimizer_id'), record.get('case_id'),
        'completed' if passed else 'failed', record.get('close_reason'), snapshots.SELECTION_POLICY,
        selected.selection_reason if selected else None, selected.snapshot_id if selected else None,
        selected.event_seq if selected else None, selected.content_sha256 if selected else None,
        selected.observation if selected else None, observation.get('kind'), observation.get('mode'),
        observation.get('latency_us'), observation.get('metric_name'), observation.get('unit'),
        'passed' if passed else None, float(run['median_effective_us']) if passed else None,
        float(run['speedup']) if passed else None, intact, usage['hardware_actions'], usage['finish_actions'],
        seq, error_code, record.get('completed_at'), message)


def _read_if_present(path):
    with file_tree.open_parent(path) as (fd, name):
        try:
            os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
    return session._read(path)


def _validate_final(root, value, record, events):
    try:
        result = FinalResult(**value)
        if (type(result.schema_version) is not int or result.schema_version != 1
                or any(getattr(result, key) != record[key] for key in ('session_id', 'optimizer_id', 'case_id', 'status', 'close_reason', 'completed_at'))
                or result.status not in {'completed', 'failed'} or record['active_action'] is not None
                or result.hardware_actions_used != record['usage']['hardware_actions']
                or result.finish_actions_used != record['usage']['finish_actions']
                or result.selection_policy != snapshots.SELECTION_POLICY
                or type(result.final_integrity_passed) is not bool
                or type(result.hardware_actions_used) is not int or type(result.finish_actions_used) is not int
                or result.finish_event_seq is not None and type(result.finish_event_seq) is not int
                or result.selected_event_seq is not None and type(result.selected_event_seq) is not int
                or result.best_observed_latency_us is not None and not actions._positive(result.best_observed_latency_us)):
            raise ValueError('identity/state/budget')
        saved = _read_if_present(root / 'final/selection.json')
        selected = snapshots.Selection(**saved) if saved else None
        if selected is None and result.selected_snapshot is not None:
            event = next(e for e in events if e['seq'] == result.selected_event_seq and e['snapshot_id'] == result.selected_snapshot)
            original = root / record['workspaces'][event['workspace']]['path']
            selected = snapshots.Selection(event['snapshot_id'], event['seq'], event['content_sha256'],
                            str(original), event['observation'], result.selection_reason)
        if result.status == 'completed' and selected != snapshots._select_best_snapshot_locked(root):
            raise ValueError('selection does not match protected history')
        expected = _result(record, selected, error_code=result.error_code, intact=result.final_integrity_passed,
                           seq=result.finish_event_seq, message=result.message)
        for key in ('selection_reason', 'selected_snapshot', 'selected_event_seq', 'selected_content_sha256',
                    'selected_observation', 'best_observed_source', 'best_observed_mode',
                    'best_observed_latency_us', 'best_observed_metric_name', 'best_observed_unit'):
            if getattr(result, key) != getattr(expected, key):
                raise ValueError('selection provenance')
        finishes = [event for event in events if event['action'] == 'finish']
        if len(finishes) != result.finish_actions_used:
            raise ValueError('finish ledger')
        if finishes and finishes[0]['seq'] != result.finish_event_seq:
            raise ValueError('finish event')
        if not finishes and result.finish_event_seq is not None:
            raise ValueError('unexpected finish event')
        if result.status == 'completed':
            if (result.error_code is not None or not result.final_integrity_passed or selected is None or len(finishes) != 1
                    or not actions._positive(result.final_verified_latency_us) or not actions._positive(result.final_verified_speedup)):
                raise ValueError('success prerequisites')
            event = finishes[0]
            raw, _ = actions.validate_result(root / 'final/authority_result.json', record, root / 'final/workspace', 'evaluate', 'clean', event['exit_code'])
            verified = _result(record, selected, raw=raw, intact=True, seq=event['seq'])
            if (event['error_code'] is not None or not event['integrity_passed'] or event['snapshot_id'] is not None
                    or event['content_sha256'] != selected.content_sha256 or event['result'] != raw
                    or any(getattr(result, key) != getattr(verified, key) for key in
                           ('final_verified_correctness', 'final_verified_latency_us', 'final_verified_speedup'))):
                raise ValueError('final authority')
        elif (not isinstance(result.error_code, str) or not result.error_code
              or any(getattr(result, key) is not None for key in
                     ('final_verified_correctness', 'final_verified_latency_us', 'final_verified_speedup'))):
            raise ValueError('failure has verified metrics')
        return result
    except (KeyError, TypeError, ValueError, StopIteration, session.SessionError) as error:
        raise integrity.InfrastructureError(f'invalid committed final result: {error}') from error


def _existing_final(root):
    with session._session_lock(root):
        record, events = session._load_locked(root)
        value = _read_if_present(root / 'final_result.json')
    return _validate_final(root, value, record, events) if value is not None else None


def _commit_failure(root, result):
    with session._session_lock(root):
        record, events = session._load_locked(root)
        if record['active_action'] is not None:
            raise integrity.InfrastructureError('active execution must be resolved before terminal publication')
        if result.selected_snapshot is None:
            saved = _read_if_present(root / 'final/selection.json')
            if saved:
                result = _result(record, snapshots.Selection(**saved), error_code=result.error_code,
                                 intact=result.final_integrity_passed, seq=result.finish_event_seq, message=result.message)
        if record['usage']['finish_actions'] and result.finish_event_seq is None:
            result.finish_event_seq = next(event['seq'] for event in events if event['action'] == 'finish')
        record.update(status='failed', exploration_closed=True, completed_at=session._now())
        result.completed_at = record['completed_at']
        result.close_reason = record['close_reason']
        session._commit(root, record, record_updates={'final_result': session._bytes(result.to_dict())})
    return result


def _prepare_final(root, record, selected):
    final = root / 'final'
    target = final / 'workspace'
    source = Path(record['snapshots_root']) / selected.snapshot_id / 'source'
    # Handled pre-reservation retries always recreate a clean source tree.
    with file_tree.open_directory(final) as fd:
        file_tree.remove_at(fd, 'workspace')
        for name in ('result.json', 'stdout.log', 'stderr.log'):
            file_tree.remove_at(fd, name)
    manifest = workspace.copy_candidate_tree(source, target)
    if manifest['content_sha256'] != selected.content_sha256:
        raise integrity.InfrastructureError('final source differs from selected Snapshot')
    for relative, entry in file_tree.scan_tree(target, reject_special=True).items():
        if entry['kind'] == 'directory':
            with file_tree.open_directory(target / relative) as fd:
                os.fchmod(fd, entry['mode'] | 0o200)
        elif entry['kind'] == 'file':
            with file_tree.open_file(target / relative) as fd:
                os.fchmod(fd, entry['mode'] | 0o200)
    case = session._read(Path(record['case_dir']) / 'case.json')
    workspace.write_context(Path(record['case_dir']), Path(record['sealed_dir']), case, target, parent_workspace=source)
    roots = {'context_final': integrity.ProtectionRoot(target / '.benchmark_context', 'context', readonly=True),
             'final_source': integrity.ProtectionRoot(final / 'source_manifest.json', 'records'),
             'final_selection': integrity.ProtectionRoot(final / 'selection.json', 'records'),
             'final_evaluator_result': integrity.ProtectionRoot(final / 'authority_result.json', 'records')}
    current = session._read(root / 'protection/manifest.json')['roots']
    for name, info in roots.items():
        if name in current and (current[name]['path'] != str(info.path) or current[name]['group'] != info.group):
            raise integrity.InfrastructureError('final protection binding changed')
    added = {name: info for name, info in roots.items() if name not in current}
    if added:
        integrity.register_roots(root / 'protection', added)
    with session._session_lock(root):
        session._load_locked(root)
        integrity.commit_records(root / 'protection', {'final_source': session._bytes(manifest),
                                'final_selection': session._bytes(selected.to_dict())})
    return manifest


def _raw_bytes(path):
    try:
        with file_tree.open_file(path) as fd, os.fdopen(os.dup(fd), 'rb') as stream:
            return stream.read()
    except FileNotFoundError:
        return None


def _finish_locked(root, record, selected, signal_state):
    manifest = _prepare_final(root, record, selected)
    seq = record['next_event_seq']
    frozen = root / 'final/workspace'
    session.reserve_action(root, seq, 'finish', workspace_path=frozen, source_manifest=manifest)
    argv = [sys.executable, str(common.BENCHMARK_ROOT / 'evaluator/evaluate.py'), '--case', record['case_id'],
            '--candidate', str(frozen), '--phase', 'all', '--profile-mode', 'clean', '--result', str(root / 'final/result.json')]
    environment = os.environ.copy()
    environment.pop('PYPTO_BENCH_ENV', None)
    environment.pop('PYPTO_BENCH_PROFILE', None)
    exit_code, error_code, message, raw = 1, None, None, None
    unsafe_stop = False
    started = False
    try:
        exit_code = actions._run_process(argv, environment, root / 'final', root, seq, signal_state)
        started = True
    except actions.ProcessCleanupError:
        unsafe_stop = True
        raise
    except (actions.ActionInterrupted, KeyboardInterrupt) as error:
        error_code, message, exit_code = 'interrupted', str(error), 130
    except Exception as error:
        error_code, message = 'infrastructure_error', str(error)
    finally:
        if not unsafe_stop:
            marker = _read_if_present(actions._marker_path())
            started = bool(marker and marker.get('process'))
            intact, repaired = actions._postcheck_action(root, frozen, manifest)
    if signal_state['received']:
        error_code, message, exit_code = 'interrupted', f"received signal {signal_state['received']}", 130
    if not intact and error_code != 'interrupted':
        error_code, message = 'integrity_failed', 'protected content or final source changed during finish'
    if error_code is None:
        try:
            raw, _ = actions.validate_result(root / 'final/result.json', record, frozen, 'evaluate', 'clean', exit_code)
        except (session.SessionError, integrity.InfrastructureError, OSError, ValueError) as error:
            error_code, message = 'action_failed', str(error)
    signal_state['active'] = False
    record = session.read_session(root)
    result = _result(record, selected, error_code=error_code, raw=raw, intact=intact, seq=seq, message=message)
    body = result.to_dict()
    raw_bytes = _raw_bytes(root / 'final/result.json')
    updates = {'final_evaluator_result': raw_bytes} if raw_bytes is not None else None
    session.complete_action(root, seq, exit_code=exit_code, error_code=error_code, integrity_passed=intact,
                            result=raw, repaired_paths=repaired, action_started=started,
                            final_result=body, record_updates=updates)
    actions._clear_marker()
    return FinalResult(**body)


def finalize_session(session_root, close_reason):
    selected = None
    root = None
    try:
        # This short update allows a currently accepted last action to finish;
        # the global lock below waits for its owner before final execution.
        session.close_exploration(session_root, close_reason)
        with session.execution_lock(), actions._signals() as signal_state:
            root = session._path(session_root, code='invalid_request')
            actions.recover_global_action()
            actions._recover_unmarked_action(root)
            record = session.read_session(root)
            snapshots._snapshot_bindings(record, session._read(root / 'protection/manifest.json'))
            actions._repair_for_action(root)
            existing = _existing_final(root)
            if existing is not None:
                return existing
            record = session.read_session(root)
            if record['usage']['finish_actions']:
                # A consumed finish without a committed final result is never
                # replayed, including a completed raw PASS from an interrupted
                # non-atomic caller. Only existing final_result proves success.
                saved = _read_if_present(root / 'final/selection.json')
                selected = snapshots.Selection(**saved) if saved else None
                with session._session_lock(root):
                    _, events = session._load_locked(root)
                finish = next((e for e in events if e['action'] == 'finish'), None)
                code = finish['error_code'] if finish and finish['error_code'] else 'interrupted'
                result = _result(record, selected, error_code=code, seq=finish['seq'] if finish else None)
                return _commit_failure(root, result)
            if record['status'] == 'failed':
                return _commit_failure(root, _result(record, error_code='infrastructure_error', message='Session has an infrastructure failure'))
            selected = snapshots._select_best_snapshot_locked(root)
            if selected is None:
                return _commit_failure(root, _result(record, error_code='no_valid_snapshot'))
            return _finish_locked(root, record, selected, signal_state)
    except (Exception, KeyboardInterrupt) as error:
        code = getattr(error, 'error_code', 'infrastructure_error')
        if code not in {'interrupted', 'invalid_request'}:
            code = 'infrastructure_error'
        record = {}
        if root is not None:
            try:
                with session.execution_lock():
                    # If the S1 manifest was already published, restore the
                    # entire terminal transaction rather than undoing success.
                    try:
                        existing = _existing_final(root)
                    except Exception:
                        existing = None
                    if existing is not None:
                        return existing
                    record = session.read_session(root)
                    if record['active_action'] is None:
                        return _commit_failure(root, _result(record, selected, error_code=code, message=str(error)))
                    if code != 'invalid_request':
                        session.fail_session(root, 'finalization_infrastructure_failure')
            except Exception:
                pass
        return _result(record, selected, error_code=code, message=str(error))
