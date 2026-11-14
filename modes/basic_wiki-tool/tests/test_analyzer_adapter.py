import copy
import hashlib
import json
import threading
from pathlib import Path

import pytest
import session
from conftest import MODE, ROOT, publish


def adapter_type():
    assert (MODE / 'analyzer_adapter.py').is_file(), 'Task 1 adapter not implemented'
    from analyzer_adapter import AnalyzerAdapter
    return AnalyzerAdapter


def test_copy_and_specs_match_source(profile_root):
    Adapter = adapter_type()
    from modes.ptomni.pto_analyzer import ProfileRef, open_toolset
    manifest = json.loads((MODE / 'analyzer_manifest.json').read_text())
    expected_paths = ['__init__.py', 'profile.py', 'toolset.py'] + [str(p.relative_to(ROOT / 'modes/ptomni/pto_analyzer')) for p in (ROOT / 'modes/ptomni/pto_analyzer/diagnostics').glob('*.py')]
    assert set(manifest['files']) == set(expected_paths)
    for relative, identity in manifest['files'].items():
        source = (ROOT / 'modes/ptomni/pto_analyzer' / relative).read_bytes()
        assert (MODE / 'pto_analyzer' / relative).read_bytes() == source
        assert identity['source_sha256'] == identity['copy_sha256'] == hashlib.sha256(source).hexdigest()
    expected = []
    ref = ProfileRef.from_event(profile_root, 1)
    for role in ('Compute', 'Dataflow', 'Schedule'):
        for spec in open_toolset(role, ref).tool_specs:
            schema = copy.deepcopy(spec['input_schema'])
            schema['properties']['profile'] = {'type': 'string', 'description': 'Session profile event ID (session_id:event_seq). Required with cursor.'}
            expected.append(dict(name=spec['name'], label=spec['name'], description=spec['description'], parameters=schema))
    assert Adapter(profile_root).specifications() == expected == manifest['specifications']
    import pto_analyzer
    assert Path(pto_analyzer.__file__).parent == MODE / 'pto_analyzer'


def test_profile_selection_lifecycle(profile_root):
    Adapter = adapter_type()
    root = profile_root
    data = (root / 'events.jsonl').read_bytes()
    (root / 'events.jsonl').write_text('')
    adapter = Adapter(root)
    assert len(adapter.specifications()) == 9
    with pytest.raises(ValueError, match='ptobench profile'):
        adapter.call('tile_access_check', {})
    (root / 'events.jsonl').write_bytes(data)
    result = adapter.call('tile_access_check', {'limit': 1})
    assert result['profile']['candidate_sha256'] == 'a' * 64
    cursor = result['details']['next_cursor']
    assert cursor
    publish(root, 2)
    assert adapter.call('tile_access_check', {})['profile']['event_id'] == 'fixture-session:2'
    assert adapter.call('tile_access_check', {'profile': 'fixture-session:1', 'cursor': cursor})['profile']['candidate_sha256'] == 'a' * 64
    assert adapter.call('tile_access_check', {'profile': 'fixture-session:1'})['profile']['event_seq'] == 1
    for bad in ['other-session:1', 'fixture-session:99', str(root), '../session', 1, None]:
        with pytest.raises(ValueError):
            adapter.call('tile_access_check', {'profile': bad})
    with pytest.raises(ValueError, match='profile'):
        adapter.call('tile_access_check', {'cursor': cursor})
    with pytest.raises(ValueError):
        adapter.call('tile_access_check', {'profile': 'fixture-session:2', 'cursor': cursor})
    publish(root, 3, case_id='another.case')
    with pytest.raises(ValueError):
        adapter.call('tile_access_check', {'profile': 'fixture-session:3'})
    assert adapter.call('tile_access_check', {})['profile']['event_seq'] == 2
    with pytest.raises(KeyError):
        adapter.call('unknown', {})


def test_profile_open_uses_consistent_session_snapshot(profile_root, monkeypatch):
    Adapter = adapter_type()
    import analyzer_adapter as module
    adapter = Adapter(profile_root)
    original = module.open_toolset
    opened = threading.Event()
    finished = threading.Event()
    results = []

    def checked_open(*args):
        assert getattr(session._LOCKS, 'session', False)
        tool = original(*args)
        class Checked:
            def call(self, *args):
                assert not getattr(session._LOCKS, 'session', False)
                return tool.call(*args)
        opened.set()
        return Checked()

    monkeypatch.setattr(module, 'open_toolset', checked_open)
    def invoke():
        try:
            results.append(adapter.call('compute_mapping_check', {}))
        finally:
            finished.set()
    with session._session_lock(profile_root):
        thread = threading.Thread(target=invoke)
        thread.start()
        assert not opened.wait(.1)
        publish(profile_root, 2)
    assert finished.wait(5)
    thread.join()
    assert results[0]['profile']['event_seq'] == 2
