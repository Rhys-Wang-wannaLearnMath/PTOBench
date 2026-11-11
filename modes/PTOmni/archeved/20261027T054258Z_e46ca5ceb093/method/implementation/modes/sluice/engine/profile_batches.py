"""Resolve a complete public fixture batch into immutable atomic evidence.

Each child owns one fixture/build. The parent never pools timing samples or
pretends that independent fixture runs share a clock or form an execution chain.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from .evidence_guard import opaque
from .profile_adapter import ProfileHandle, read_json, within


def register_batch(adapter, obs, raw, root, parent, workspace):
    public = adapter.context.public_context_root(workspace)
    declared = read_json(public/'case.json').get('fixture_ids')
    runs = raw.get('runs')
    if (not isinstance(declared, list) or not 2 <= len(declared) <= 64
            or any(not isinstance(v, str) or not re.fullmatch(r'fixture_[0-9]{3,6}', v) for v in declared)
            or len(set(declared)) != len(declared) or not isinstance(runs, list)
            or len(runs) != len(declared) or any(not isinstance(r, dict) or r.get('passed') is not True for r in runs)
            or {r.get('fixture_id') for r in runs} != set(declared)):
        raise ValueError('fixture_batch_incomplete')
    manifest = raw.get('build_manifest')
    if (not isinstance(manifest, list) or len(manifest) != len(declared)
            or any(not isinstance(b, dict) or b.get('invocation') != 0 or type(b.get('invocation')) is not int
                   for b in manifest)
            or {b.get('fixture_id') for b in manifest} != set(declared)):
        raise ValueError('fixture_build_manifest_incomplete')
    builds = {b['fixture_id']:within(Path(b['build_dir']), root/'build_output') for b in manifest}
    if len(set(builds.values())) != len(declared) or any(not b.is_dir() for b in builds.values()):
        raise ValueError('fixture_build_manifest_ambiguous')
    original = root/'latest_result.json'
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    directory = adapter.context.work_root/'profile_inputs'/parent.ref.split(':')[1]
    directory.mkdir(parents=True, exist_ok=True)
    children = []
    for run in sorted(runs, key=lambda r:r['fixture_id']):
        fixture = run['fixture_id']
        build = builds[fixture]
        payload = {**raw, 'runs':[run], 'build_dir':str(build),
                   'build_manifest':[b for b in manifest if b['fixture_id'] == fixture],
                   'batch_projection':{'source_sha256':original_sha, 'fixture_id':fixture}}
        source = directory/(fixture+'.json')
        text = json.dumps(payload, sort_keys=True, allow_nan=False)
        if source.exists() and source.read_text() != text:
            raise ValueError('fixture_projection_changed')
        source.write_text(text)
        child = ProfileHandle(opaque('profile', (parent.ref, fixture)), parent.identity,
                              obs.method_action_id, obs.bench_event_seq, obs.result_path)
        if parent.identity.profile_mode == 'clean':
            child = adapter._register_clean_payload(obs, payload, root, child, source, workspace)
        else:
            child = adapter._register_profile_payload(obs, payload, root, child, build)
        if child.availability != 'available':
            raise ValueError('fixture_capture_incomplete')
        # Anchor the complete original result as well as the normalized child.
        # A generated per-fixture JSON is never sufficient provenance by itself.
        from profile_db.ingest.analysis import register_paths
        def anchor(handle):
            if handle.member_refs:
                members = [anchor(adapter.handles[r]) for r in handle.member_refs]
                return adapter._save(replace(handle, artifact_fingerprint=tuple(c.artifact_fingerprint for c in members)))
            copied = []
            try:
                with adapter.db._writing() as conn:
                    register_paths(conn, handle.db_run_id, {'fixture_batch_result':original},
                                   copy=adapter.db.path is not None, copied=copied)
            except Exception:
                for p in copied:
                    p.unlink(missing_ok=True)
                raise
            internal = adapter.db.get_profile_handle(handle.db_run_id)
            return adapter._save(replace(handle, artifact_fingerprint=tuple(internal['artifact_fingerprint'])))
        children.append(anchor(child))
    if hashlib.sha256(original.read_bytes()).hexdigest() != original_sha:
        raise ValueError('fixture_batch_changed')
    return adapter._save(replace(parent, identity=replace(children[0].identity, fixture_id=None, build_ref=None),
        db_run_id=children[0].db_run_id, artifact_fingerprint=tuple(c.artifact_fingerprint for c in children),
        availability='available', member_refs=tuple(c.ref for c in children), member_kind='fixture',
        member_slots=tuple(c.identity.fixture_id for c in children)))


def validate_original(db, handle):
    """Revalidate diagnostic originals and normalized clean results on every read."""
    from .device_identity import read_identity
    read_identity(db, handle)
    from ..profiling.diagnostic_integrity import validate
    validate(db, handle)
    from profile_db.runtime_adapter.storage import resolve
    present = db.connection.execute(
        "SELECT 1 FROM artifact WHERE run_id=? AND kind='fixture_batch_result'", [handle.db_run_id]).fetchone()
    if not present:
        return
    original = resolve(db.connection, handle.db_run_id, 'fixture_batch_result')
    if handle.identity.profile_mode != 'clean':
        return
    projected = read_json(resolve(db.connection, handle.db_run_id, 'clean_result'))
    binding = projected.get('batch_projection', {})
    raw = read_json(original)
    rows = [r for r in raw.get('runs', []) if r.get('fixture_id') == handle.identity.fixture_id]
    if (binding.get('source_sha256') != hashlib.sha256(original.read_bytes()).hexdigest()
            or binding.get('fixture_id') != handle.identity.fixture_id or rows != projected.get('runs')):
        raise ValueError('fixture_projection_changed')
