"""Publish successful frozen Candidate trees, with immutable protection (S3)."""
from __future__ import annotations
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import uuid

import file_tree
import integrity
import session
import workspace

SELECTION_POLICY = 'lowest_observed_latency_else_latest_pass'


@dataclass
class Selection:
    snapshot_id: str
    event_seq: int
    content_sha256: str
    source_workspace: str
    observation: dict
    selection_reason: str
    selection_policy: str = SELECTION_POLICY

    def to_dict(self):
        return asdict(self)


def _snapshot_bindings(record, manifest):
    import re
    for name, info in manifest['roots'].items():
        if info['group'] == 'snapshots':
            if (not re.fullmatch(r'snapshot_\d{6,}', name) or not info['readonly']
                    or info['path'] != str(Path(record['snapshots_root']) / name)):
                raise integrity.InfrastructureError('Snapshot protection binding is invalid')


def _select_best_snapshot_locked(session_root):
    import session_actions
    session._require_execution()
    record = session.read_session(session_root)
    root = Path(record['session_root'])
    protection = session._read(root / 'protection/manifest.json')
    _snapshot_bindings(record, protection)
    session_actions._repair_for_action(root)
    with session._session_lock(root):
        record, events = session._load_locked(root)
    candidates = []
    for event in events:
        sid = event['snapshot_id']
        if (event['action'] not in {'evaluate', 'profile'} or not sid
                or event['error_code'] is not None or event['exit_code'] != 0
                or event.get('state') != 'completed'
                or not event['hardware_action'] or not event['action_started'] or not event['integrity_passed']):
            continue
        expected_id = f"snapshot_{event['seq']:06d}"
        if sid != expected_id:
            raise integrity.InfrastructureError('Snapshot/event sequence mismatch')
        info = protection['roots'].get(sid)
        if info is None:
            continue  # Published but unregistered directories are not candidates.
        if info['group'] != 'snapshots':
            raise integrity.InfrastructureError('Snapshot is not protected as a Snapshot')
        snapshot = Path(record['snapshots_root']) / sid
        metadata = session._read(snapshot / 'metadata.json')
        action = root / 'staging' / f"action_{event['seq']:06d}"
        for label, filename in ((f"action_source_{event['seq']}", 'source_manifest.json'),
                                (f"action_result_{event['seq']}", 'authority_result.json')):
            binding = protection['roots'].get(label, {})
            if binding.get('group') != 'records' or binding.get('path') != str(action / filename):
                raise integrity.InfrastructureError('Snapshot authority is not protected')
        source = session._read(action / 'source_manifest.json')
        session._validate_source_manifest(source)
        raw, observation = session_actions.validate_result(action / 'authority_result.json', record,
                               action / 'workspace', event['action'],
                               'clean' if event['action'] == 'evaluate' else session.PUBLIC_MODES.get(event['mode']), 0)
        original = root / record['workspaces'][event['workspace']]['path']
        expected = {'schema_version': 1, 'snapshot_id': sid, 'session_id': record['session_id'],
                    'case_id': record['case_id'], 'optimizer_id': record['optimizer_id'],
                    'source_workspace': str(original), 'event_seq': event['seq'],
                    'content_sha256': source['content_sha256'], 'observation': observation}
        if (metadata != expected or type(metadata.get('schema_version')) is not int or type(metadata.get('event_seq')) is not int
                or event.get('observation') != observation or event['result'] != raw
                or event['content_sha256'] != source['content_sha256']
                or workspace.source_manifest(snapshot / 'source') != source):
            raise integrity.InfrastructureError('Snapshot source/metadata/authority disagree')
        candidates.append(Selection(sid, event['seq'], source['content_sha256'], str(original), observation, ''))
    if not candidates:
        return None
    timed = [candidate for candidate in candidates if session_actions._positive(candidate.observation['latency_us'])]
    selected = min(timed, key=lambda c: (c.observation['latency_us'], c.event_seq)) if timed else max(candidates, key=lambda c: c.event_seq)
    selected.selection_reason = 'lowest_observed_latency' if timed else 'latest_pass_without_latency'
    return selected


def select_best_snapshot(session_root):
    """Recover and deterministically select protected observations; no hardware."""
    import session_actions
    with session.execution_lock():
        session_actions.recover_global_action()
        session_actions._recover_unmarked_action(session_root)
        try:
            return _select_best_snapshot_locked(session_root)
        except (session.SessionError, workspace.InvalidWorkspaceError, KeyError, TypeError, ValueError) as error:
            raise integrity.InfrastructureError(f'invalid Snapshot history: {error}') from error


def finalize_session(session_root, close_reason):
    """Internal S4 entry; deliberately absent from the five-operation CLI."""
    from finalize import finalize_session as implementation
    return implementation(session_root, close_reason)


def create_snapshot(session_root, frozen_source, event):
    session._require_execution()
    record = session.read_session(session_root)
    if event.get('correctness') != 'PASS' or not event.get('integrity_passed') or event.get('exit_code') != 0:
        raise session.SessionError('only successful, intact observations can create a Snapshot')
    seq = event['seq']
    snapshot_id = f'snapshot_{seq:06d}'
    parent = session._path(record['snapshots_root'])
    target = parent / snapshot_id
    temporary = parent / f'.{snapshot_id}.{uuid.uuid4().hex}'
    try:
        with file_tree.open_directory(parent) as fd:
            try:
                os.stat(snapshot_id, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise integrity.InfrastructureError('Snapshot ID already exists')
        with file_tree.open_directory(temporary, create=True):
            pass
        manifest = workspace.copy_candidate_tree(frozen_source, temporary / 'source')
        if manifest['content_sha256'] != event['content_sha256']:
            raise integrity.InfrastructureError('Snapshot source differs from accepted frozen source')
        metadata = {
            'schema_version': 1, 'snapshot_id': snapshot_id,
            'session_id': record['session_id'], 'case_id': record['case_id'], 'optimizer_id': record['optimizer_id'],
            'source_workspace': event['workspace'], 'event_seq': seq,
            'content_sha256': manifest['content_sha256'], 'observation': event['observation'],
        }
        file_tree.atomic_bytes(temporary / 'metadata.json', session._bytes(metadata), mode=0o444)
        entries = file_tree.scan_tree(temporary / 'source', reject_special=True)
        for relative, entry in entries.items():
            path = temporary / 'source' / relative
            if entry['kind'] == 'file':
                with file_tree.open_file(path) as fd:
                    os.fchmod(fd, 0o444)
        for relative, entry in sorted(entries.items(), key=lambda item: item[0].count('/'), reverse=True):
            if entry['kind'] == 'directory':
                with file_tree.open_directory(temporary / 'source' / relative) as fd:
                    os.fchmod(fd, 0o555)
        with file_tree.open_directory(temporary) as fd:
            os.fchmod(fd, 0o555)
        with file_tree.open_directory(parent) as fd:
            os.rename(temporary.name, target.name, src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
        if session._read(target / 'metadata.json') != metadata or workspace.source_manifest(target / 'source') != manifest:
            raise integrity.InfrastructureError('published Snapshot does not match its metadata/source')
        integrity.register_roots(Path(session_root) / 'protection', {
            f'snapshot_{seq:06d}': integrity.ProtectionRoot(target, 'snapshots', readonly=True),
        })
        return snapshot_id
    finally:
        with file_tree.open_directory(parent) as fd:
            file_tree.remove_at(fd, temporary.name)
