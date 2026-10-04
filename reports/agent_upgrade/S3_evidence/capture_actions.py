"""Archive small fake-CPU action evidence; no real Case/hardware is executed."""
import json
from pathlib import Path
import shutil
import sys

repo = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(repo / 'benchmarks/pypto_lib_bench-v0.2/evaluator/tests'))
from action_test_support import ActionTestBase

destination = Path(__file__).parent / 'fake_actions'
destination.mkdir()  # Evidence is never overwritten by a later run.
fixture = ActionTestBase('runTest')
try:
    fixture.setUp()
    outcomes = []
    requests = [(['evaluate'], 0), (['profile', 'runtime'], 0), (['profile', 'pmu'], 0),
                (['profile', 'compiler'], 0), (['fork'], 0), (['profile', 'hardware'], 2)]
    for index, (argv, expected) in enumerate(requests, 1):
        process, result = fixture.cli(argv)
        assert process.returncode == expected, (argv, result)
        outcomes.append({'argv': argv, 'exit_code': process.returncode, 'result': result})
        (destination / f'cli_{index:02d}.stdout').write_bytes(process.stdout)
        (destination / f'cli_{index:02d}.stderr').write_bytes(process.stderr)
    fixture.configure(fail=True)
    process, result = fixture.cli(['evaluate'])
    assert process.returncode == 1 and result['snapshot_id'] is None, result
    outcomes.append({'argv': ['evaluate'], 'exit_code': process.returncode, 'result': result})
    (destination / 'cli_07.stdout').write_bytes(process.stdout)
    (destination / 'cli_07.stderr').write_bytes(process.stderr)
    (destination / 'outcomes.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    for name in ('session.json', 'events.jsonl'):
        shutil.copyfile(fixture.handle.session_root / name, destination / name)
    shutil.copyfile(fixture.counter, destination / 'evaluator_calls.jsonl')
    shutil.copytree(fixture.handle.snapshots_root, destination / 'snapshots')
    shutil.copytree(fixture.handle.initial_workspace / '.benchmark_context', destination / 'context')
    actions = destination / 'actions'
    actions.mkdir()
    for action in (fixture.handle.session_root / 'staging').iterdir():
        saved = actions / action.name
        saved.mkdir()
        for name in ('source_manifest.json', 'authority_result.json', 'result.json', 'stdout.log', 'stderr.log'):
            source = action / name
            if source.exists():
                shutil.copyfile(source, saved / name)
    print('fake CPU evidence: 7 CLI operations, 5 Evaluator invocations, 4 Snapshots; no hardware')
finally:
    fixture.doCleanups()
