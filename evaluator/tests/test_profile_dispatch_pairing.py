from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

BENCHMARK = Path(__file__).resolve().parents[2]
SHARED = BENCHMARK / 'modes/_shared'
sys.path[:0] = [str(BENCHMARK), str(SHARED)]

SPEC = importlib.util.spec_from_file_location(
    'profile_dispatch_pairing_test', SHARED / 'profile_dispatch_pairing.py')
assert SPEC is not None and SPEC.loader is not None
pairing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pairing)


def identity(*, run_id, capture_index, task_slot=0):
    return {
        'schema_version': 1,
        'run_id': run_id,
        'task_slot': task_slot,
        'group_index': 0,
        'group_size': 1,
        'chip_rank': 0,
        'local_capture_index': capture_index,
        'endpoint_dispatch_id': run_id,
        'pipeline_slot': 0,
        'pipeline_generation': run_id,
        'callable_digest': 'a' * 64,
    }


class DispatchPairingTest(unittest.TestCase):
    def make_capture(self, root, *, run_id, capture_index, filename, data):
        directory = root / f'rank0/d{capture_index}'
        directory.mkdir(parents=True)
        (directory / 'dispatch_identity.json').write_text(json.dumps(
            identity(run_id=run_id, capture_index=capture_index)))
        path = directory / filename
        path.write_bytes(data)
        return directory, path

    def test_pairs_dispatches_when_pass_run_ids_are_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = b'{"tasks":[{"kernel_ids":[-1,-1,-1]}],"edges":[]}'
            graph_dir, _ = self.make_capture(
                root, run_id=1, capture_index=0, filename='deps.json', data=graph)
            _, timing_path = self.make_capture(
                root, run_id=1, capture_index=1,
                filename='chip_swimlane_records.json', data=b'{"aicore_tasks":[]}')

            result = pairing.join(root, [(graph_dir, graph,
                                          identity(run_id=1, capture_index=0))])

            self.assertEqual(result, 1)
            self.assertEqual((timing_path.parent / 'deps.json').read_bytes(), graph)

    def test_rejects_unmatched_dispatch_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = b'{"tasks":[],"edges":[]}'
            graph_dir, _ = self.make_capture(
                root, run_id=1, capture_index=0, filename='deps.json', data=graph)
            _, _ = self.make_capture(
                root, run_id=2, capture_index=1,
                filename='chip_swimlane_records.json', data=b'{"aicore_tasks":[]}')
            graph_identity = identity(run_id=1, capture_index=0)
            graph_identity['task_slot'] = 4

            with self.assertRaisesRegex(ValueError, 'unmatched or ambiguous'):
                pairing.join(root, [(graph_dir, graph, graph_identity)])

    def test_resets_mutable_state_between_l3_runtime_passes(self):
        from pypto.runtime import distributed_runner
        import golden.runner as golden_runner

        events = []

        def sdk_two_pass(_dc, _config, _base, run_pass):
            run_pass(SimpleNamespace(enable_dep_gen=True, enable_chip_swimlane=0))
            run_pass(SimpleNamespace(enable_dep_gen=False, enable_chip_swimlane=4))

        def observe(pass_config):
            events.append('graph' if pass_config.enable_dep_gen else 'timing')

        with mock.patch.object(distributed_runner, '_run_l3_swimlane_two_pass', sdk_two_pass), \
                mock.patch.object(pairing, 'snapshot', side_effect=lambda _root: events.append('snapshot') or []), \
                mock.patch.object(pairing, 'join') as join:
            with golden_runner.l3_pass_reset(lambda: events.append('reset')):
                with pairing.capture(enabled=True):
                    distributed_runner._run_l3_swimlane_two_pass(
                        None, None, Path('.'), observe,
                    )

        self.assertEqual(events, ['graph', 'snapshot', 'reset', 'timing'])
        join.assert_called_once()


if __name__ == '__main__':
    unittest.main()
