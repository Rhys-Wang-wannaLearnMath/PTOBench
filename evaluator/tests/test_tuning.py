"""Configuration-only checks: native dispatch and actual compilation are forbidden."""
import contextlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

EVALUATOR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EVALUATOR), str(EVALUATOR / 'tests')]
import common
import tuning
import workflow
import profile_modes
import inventory_tuning
from action_test_support import ActionTestBase


class TuningValidationTests(unittest.TestCase):
    def test_defaults_and_every_public_combination(self):
        self.assertEqual(common.TUNING_DEFAULTS, tuning.DEFAULTS)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(common.read_tuning(root), common.TUNING_DEFAULTS)
            for planner, analyze in itertools.product(('PYPTO', 'DSA_RP', 'PTOAS'), (False, True)):
                value = {'memory_planner': planner, 'analyze_auto_scopes_for_deps': analyze}
                (root / 'tuning.json').write_text(json.dumps(value))
                self.assertEqual(common.read_tuning(root), value)

    def test_managed_diagnostic_unknown_and_invalid_values(self):
        values = [{k: None} for k in (*tuning.MANAGED_DEFAULTS, 'pass_options', *tuning.DIAGNOSTICS)]
        values += [{'unknown': 1}, {'memory_planner': []}, {'memory_planner': 'BAD'},
                   {'memory_planner': None}, {'analyze_auto_scopes_for_deps': 1},
                   {'analyze_auto_scopes_for_deps': 'false'}, [], None]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in values:
                with self.subTest(value=value):
                    (root / 'tuning.json').write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        common.read_tuning(root)

    def test_duplicate_oversized_and_symlink_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'tuning.json'
            for payload in ('{"memory_planner":"PYPTO","memory_planner":"PTOAS"}',
                            '{', ' ' * 65537 + '{}'):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    common.read_tuning(root)
            path.unlink()
            target = root / 'target.json'
            target.write_text('{}')
            path.symlink_to(target)
            with self.assertRaises(ValueError):
                common.read_tuning(root)

    def test_capabilities_describe_inheritance_without_granting_transport_permissions(self):
        cap = tuning.capabilities({'case_id': 'test.case', 'runtime': {'ring_heap': 4096}},
                                  {'runtime': {'ring_heap': 1024, 'ring_task_window': 256}})
        self.assertEqual(cap['options']['ring_heap']['inherited_value'], 4096)
        self.assertEqual(cap['options']['ring_task_window']['inherited_value'], 256)
        self.assertEqual([n for n, v in cap['options'].items() if v['editable']], list(tuning.DEFAULTS))
        self.assertFalse(cap['hardware_tested_in_this_change'])

    def test_missing_and_silently_ignored_compile_adapter_are_rejected(self):
        for runtime in (object(), SimpleNamespace(compile_kwargs=lambda: {}),
                        SimpleNamespace(compile_kwargs=lambda: {'memory_planner':'PYPTO', 'analyze_auto_scopes_for_deps':0})):
            with self.assertRaises(ValueError):
                tuning.check_projection(runtime, tuning.DEFAULTS)


class PerCasePolicyTests(unittest.TestCase):
    def case(self, identifier):
        return common.case_paths(identifier)[2]

    def test_policies_cover_inventory_and_have_different_controls(self):
        data = json.loads(tuning.POLICY_PATH.read_text())['cases']
        cases = [json.loads(p.read_text()) for p in common.formal_case_configs()]
        self.assertEqual(set(data), {case['case_id'] for case in cases})
        self.assertGreater(len({tuple(entry['defaults']) for entry in data.values()}), 1)
        for case in cases:
            defaults = common.tuning_defaults(case)
            self.assertEqual(tuning.validate({}, case), defaults)
            self.assertEqual(defaults['memory_planner'], 'PYPTO')
            self.assertIs(defaults['analyze_auto_scopes_for_deps'], False)
            entry = data[case['case_id']]
            if entry['distributed_path']:
                self.assertNotIn('aicpu_thread_num', defaults)

    def test_unsupported_and_out_of_bounds_settings_are_rejected_per_case(self):
        case = self.case('deepseek_v4.hc_post_demo')
        for value in (True, 4, 8192, 65536, 16384.0, None, [16384]*4):
            with self.subTest(window=value), self.assertRaises(ValueError):
                tuning.validate({'ring_task_window': value}, case)
        for value in (True, 0, 1, 5, 4.0, None):
            with self.subTest(threads=value), self.assertRaises(ValueError):
                tuning.validate({'aicpu_thread_num': value}, case)
        for identifier, entry in json.loads(tuning.POLICY_PATH.read_text())['cases'].items():
            if entry['distributed_path']:
                with self.assertRaises(ValueError):
                    tuning.validate({'aicpu_thread_num': 2}, self.case(identifier))
                break

    def test_configured_controls_reach_dispatch_in_all_profiles(self):
        from pypto.runtime import runner
        case = self.case('deepseek_v4.hc_post_demo')
        value = {**common.tuning_defaults(case), 'ring_task_window': 32768, 'aicpu_thread_num': 2}
        for mode in ('clean', *profile_modes.ENABLED_PROFILE_MODES):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {'PYPTO_BENCH_PROFILE': mode}):
                _, runtime = common.environment(case, None, tuning=value)
                # Real native CallConfig construction; no Worker, compile or NPU call.
                native = runner._build_call_config(runtime, runtime_config={})
                self.assertEqual(native.aicpu_thread_num, 2)
                self.assertEqual(list(native.runtime_env.ring_task_window), [32768]*4)
                self.assertEqual(runtime._ptobench_tuning_record['requested'], value)
                self.assertEqual(runtime._ptobench_tuning_record['resolved_run_config']['ring_task_window'], 32768)

    def test_reachability_follows_export_and_excludes_unused_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root/'seed'
            seed.mkdir()
            (seed/'candidate.py').write_text('from helper import entry\n')
            (seed/'helper.py').write_text('''import pypto.language as pl
@pl.jit
def entry(x):
    return x
def unused(x):
    for i in pl.spmd(32):
        x = pl.matmul(x, x)
    return x
''')
            features = inventory_tuning.source_features(root, {'workspace':'seed', 'module':'candidate.py',
                                                              'entry':'entry', 'case_id':'test.case'})
            self.assertFalse(features['multiple_tasks_possible'])
            self.assertFalse(features['matmul_present'])
            self.assertIn('helper.entry', features['reachable_definitions'])


class BeforeHardwareTests(ActionTestBase):
    def test_case_restriction_is_checked_before_hardware_budget(self):
        import session
        entry = tuning.policy()
        entry['options']['analyze_auto_scopes_for_deps']['editable'] = False
        entry['options']['analyze_auto_scopes_for_deps']['values'] = [False]
        common.write_json(self.bench/'evaluator/tuning_policies.json', {'cases':{self.case_id:entry}})
        self.handle = self.create()
        self.env['PTOBENCH_SESSION_ROOT'] = str(self.handle.session_root)
        (self.handle.initial_workspace/'candidate/tuning.json').write_text(json.dumps({'analyze_auto_scopes_for_deps':True}))
        process, result = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 2, result)
        self.assertEqual(result['error_code'], 'invalid_request', result)
        self.assertEqual(self.calls(), [])
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'], 0)

    def test_final_acceptance_keeps_selected_snapshot_tuning(self):
        import snapshots
        chosen = {'memory_planner':'PYPTO','analyze_auto_scopes_for_deps':True}
        path = self.handle.initial_workspace/'candidate/tuning.json'
        path.write_text(json.dumps(chosen))
        process, result = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 0, result)
        path.write_text(json.dumps(common.TUNING_DEFAULTS))
        with mock.patch.dict(os.environ,self.env):
            final = snapshots.finalize_session(self.handle.session_root, 'optimizer_exit')
        self.assertEqual(final.status, 'completed', final)
        self.assertEqual(common.read_json(self.handle.session_root/'final/workspace/candidate/tuning.json'),chosen)

    def test_invalid_requests_do_not_launch_evaluator_or_consume_budget(self):
        path = self.handle.initial_workspace / 'candidate' / 'tuning.json'
        for value in ({'ring_task_window': 256}, {'ring_heap': 1073741824},
                      {'strategy': 'Default'}, {'enable_dep_gen': True},
                      {'analyze_auto_scopes_for_deps': 1}, {'unknown': 0}):
            with self.subTest(value=value):
                path.write_text(json.dumps(value))
                process, result = self.cli(['evaluate'])
                self.assertEqual(result['status'], 'REJECTED', result)
                self.assertEqual(result['error_code'], 'invalid_request', result)
                self.assertEqual(process.returncode, 2)
        self.assertEqual(self.calls(), [])
        import session
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'], 0)

    def test_final_acceptance_freezes_runtime_controls_with_compiler_controls(self):
        import snapshots
        from session_test_support import write_json
        entry = tuning.policy()
        for name, default, values in (('ring_task_window', 16384, [16384, 32768]),
                                      ('aicpu_thread_num', 4, [2, 3, 4])):
            entry['defaults'][name] = default
            entry['options'][name] = {'editable': True, 'default': default,
                                      'stage': 'dispatch', 'values': values}
        write_json(self.bench/'evaluator/tuning_policies.json', {'cases': {self.case_id: entry}})
        policy_patch = mock.patch.object(tuning, 'POLICY_PATH', self.bench/'evaluator/tuning_policies.json')
        policy_patch.start()
        self.addCleanup(policy_patch.stop)
        baseline_path = self.bench/'sealed_assets/test/case/ref_performance.json'
        baseline = common.read_json(baseline_path)
        baseline['compiler_tuning'] = entry['defaults']
        write_json(baseline_path, baseline)
        self.handle = self.create(budget=20)
        self.env['PTOBENCH_SESSION_ROOT'] = str(self.handle.session_root)
        chosen = {**entry['defaults'], 'analyze_auto_scopes_for_deps': True,
                  'ring_task_window': 32768, 'aicpu_thread_num': 2}
        path = self.handle.initial_workspace/'candidate/tuning.json'
        path.write_text(json.dumps(chosen))
        process, result = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 0, result)
        path.write_text(json.dumps(entry['defaults']))
        with mock.patch.dict(os.environ, self.env):
            final = snapshots.finalize_session(self.handle.session_root, 'optimizer_exit')
        self.assertEqual(final.status, 'completed', final)
        self.assertEqual(common.read_json(self.handle.session_root/'final/workspace/candidate/tuning.json'), chosen)

    def test_old_session_cannot_restore_old_tuning_code(self):
        import session
        import integrity
        marker = self.bench / 'evaluator/tuning.py'
        data = marker.read_bytes()
        marker.unlink()
        try:
            old = self.create()
        finally:
            marker.write_bytes(data)
        with session.execution_lock():
            with self.assertRaisesRegex(integrity.InfrastructureError, 'different tuning policy'):
                session.repair_assets(old.session_root)
        self.assertEqual(marker.read_bytes(), data)


class InstalledRuntimeTests(unittest.TestCase):
    """Use real installed configuration classes; replace the native boundary."""
    @classmethod
    def setUpClass(cls):
        try:
            from pypto.runtime import RunConfig
            from pypto.runtime import runner, distributed_runner
            from pypto.ir.pass_manager import OptimizationStrategy
        except ImportError as error:
            raise unittest.SkipTest(f'installed PyPTO required: {error}')
        cls.RunConfig = RunConfig
        cls.runner = runner
        cls.distributed_runner = distributed_runner
        cls.Strategy = OptimizationStrategy

    def test_compile_controls_reach_program_and_jit_compile_boundaries(self):
        import golden.runner as golden
        import pypto.ir
        for planner, analyze in itertools.product(('PYPTO', 'DSA_RP', 'PTOAS'), (False, True)):
            value = {'memory_planner': planner, 'analyze_auto_scopes_for_deps': analyze}
            case = {'case_id': 'test.case', 'environment': 'a2a3_single_card'}
            with mock.patch.dict(os.environ, {'PYPTO_BENCH_PROFILE':'clean'}):
                _, runtime = common.environment(case, None, tuning=value)
            tuning.check_projection(runtime, value)
            program = object()
            compile_step, _, _ = golden._program_entry(program, [])
            with mock.patch.object(pypto.ir, 'compile', return_value='compiled') as compile_spy:
                self.assertEqual(compile_step(runtime, None), 'compiled')
            self.assertEqual(compile_spy.call_args.kwargs['memory_planner'].name, planner)
            self.assertIs(compile_spy.call_args.kwargs['analyze_auto_scopes_for_deps'], analyze)
            jit = SimpleNamespace(compile=mock.Mock(return_value='compiled'))
            compile_step, _, _ = golden._jit_entry(jit, [])
            self.assertEqual(compile_step(runtime, {}), 'compiled')
            self.assertIs(jit.compile.call_args.kwargs['config'], runtime)
        self.assertEqual([s.name for s in self.Strategy], ['Default'])

    def test_l2_l3_ring_forwarding_and_distributed_thread_difference(self):
        def call_config():
            return SimpleNamespace(runtime_env=SimpleNamespace(ring_task_window=0, ring_heap=0, ring_dep_pool=0))
        from pypto.ir.distributed_compiled_program import DistributedConfig
        runtime = self.RunConfig(platform='a2a3', ring_task_window=[128, 256, 0, 0],
                                 ring_heap=1048576, ring_dep_pool=1024, aicpu_thread_num=4)
        fake = SimpleNamespace(CallConfig=call_config)
        with mock.patch.dict(sys.modules, {'pypto.runtime.task_interface': fake, 'simpler.task_interface':fake}):
            l2 = self.runner._build_call_config(runtime, runtime_config={})
            l3 = self.distributed_runner._make_call_config(DistributedConfig(device_ids=[0,1], aicpu_thread_num=2), runtime)
        self.assertEqual(l2.aicpu_thread_num, 4)
        self.assertEqual(l3.aicpu_thread_num, 2)
        for cfg in (l2,l3):
            self.assertEqual(cfg.runtime_env.ring_task_window, [128,256,0,0])
            self.assertEqual(cfg.runtime_env.ring_heap, 1048576)
            self.assertEqual(cfg.runtime_env.ring_dep_pool, 1024)

    def test_defaults_and_profile_settings_preserve_inherited_case_configuration(self):
        case = {'case_id': 'test.case', 'environment': 'a2a3_single_card',
                'runtime': {'ring_heap':1048576, 'ring_task_window':256, 'ring_dep_pool':1024}}
        for mode in ('clean', *profile_modes.ENABLED_PROFILE_MODES):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {'PYPTO_BENCH_PROFILE':mode}):
                _, runtime = common.environment(case, None)
                self.assertEqual(runtime.memory_planner.name,'PYPTO')
                self.assertIs(runtime.analyze_auto_scopes_for_deps,False)
                self.assertEqual(runtime.strategy.name,'Default')
                self.assertEqual(runtime.ring_heap,1048576)
                self.assertEqual(runtime.ring_task_window,256)
                self.assertEqual(runtime.ring_dep_pool,1024)
                for field,value in profile_modes.settings(mode).items():
                    self.assertEqual(getattr(runtime,field),value)


if __name__ == '__main__':
    unittest.main()
