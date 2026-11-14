import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

MODE = Path(__file__).resolve().parents[1]
ROOT = MODE.parents[1]
sys.path[:0] = [str(MODE), str(ROOT), str(ROOT / 'evaluator'), str(ROOT / 'evaluator/tests')]
from modes.ptomni.pto_analyzer.tests.test_toolset import fixture_session


@pytest.fixture
def profile_root(tmp_path):
    root = tmp_path / 'session'
    fixture_session(root)
    return root


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.name != '.session.lock'}


def publish(root, number, *, case_id='fixture.case', digest='b' * 64):
    source = root / 'staging/action_000001/artifacts'
    target = root / f'staging/action_{number:06d}/artifacts'
    shutil.copytree(source, target)
    index = json.loads((target / 'index.json').read_text())
    index['event_seq'] = number
    (target / 'index.json').write_text(json.dumps(index))
    event = json.loads((root / 'events.jsonl').read_text().splitlines()[0])
    event.update(seq=number, attempt_id=number, case_id=case_id, content_sha256=digest,
                 artifact_index=str((target / 'index.json').relative_to(root)),
                 profile_artifacts_path=str(target.relative_to(root)), workspace='worker_001')
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    manifest = json.loads((root / 'protection/manifest.json').read_text())
    manifest['roots']['events']['entries']['']['sha256'] = hashlib.sha256((root / 'events.jsonl').read_bytes()).hexdigest()
    manifest['roots'][f'action_artifacts_{number}'] = {'entries': {
        str(p.relative_to(target)): {'kind': 'file', 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in target.rglob('*') if p.is_file()}}
    (root / 'protection/manifest.json').write_text(json.dumps(manifest))
