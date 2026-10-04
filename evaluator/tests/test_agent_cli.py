from pathlib import Path
import json
import os
import sys

from action_test_support import ActionTestBase, SOURCE_BENCHMARK
import common
import session


class AgentCliTest(ActionTestBase):
    def test_five_operations_default_workspace_fixed_modes_and_null_profiles(self):
        for args, phase, internal in ((['evaluate'],'all','clean'), (['profile','runtime'],'correctness','runtime'),
                                    (['profile','pmu'],'correctness','hardware'), (['profile','compiler'],'correctness','compile')):
            with self.subTest(args=args):
                process, result = self.cli(args)
                self.assertEqual(process.returncode, 0, process.stderr.decode())
                self.assertEqual(result['status'], 'PASS', result)
                call=self.calls()[-1]
                self.assertEqual((call['phase'],call['profile_mode']), (phase,internal))
                self.assertEqual(call['case'],self.case_id)
                self.assertNotEqual(call['candidate'],str(self.handle.initial_workspace))
                self.assertEqual(result['latency_us'], 10.0 if internal=='clean' else None)
                self.assertTrue((self.snapshot(result)/'source/helper.py').is_file())
                authority=self.events()[-1]['result']
                self.assertIsNone(authority['bootstrap_marker'])
                self.assertIsNone(authority['inherited_profile'])
        process, result=self.cli(['fork'])
        self.assertEqual(process.returncode,0,result)
        self.assertEqual(Path(result['workspace']).name,'worker_001')
        self.assertIsNone(result['snapshot_id'])
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],4)
        self.assertEqual(len(self.calls()),4)

    def test_rejects_old_parser_modes_parameters_and_commands_without_execution(self):
        requests = [['profile','hardware'],['profile','compile'],['evaluate','--phase','all'],
                    ['evaluate','--candidate','.'],['evaluate','--result','result.json'],['evaluate','--workspace','.'],
                    ['fork','--out','x'],['fork','--name','x'],['init','test.case'],['status'],['prepare-ref','test.case'],['finalize']]
        for args in requests:
            with self.subTest(args=args):
                process,result=self.cli(args)
                self.assertEqual(process.returncode,2,result)
                self.assertEqual(result['error_code'],'invalid_request')
                self.assertEqual(result['status'],'REJECTED')
        self.assertEqual(len(self.calls()),0)
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],0)
        self.assertEqual(len(self.events()),len(requests))
        self.assertFalse(list(self.handle.snapshots_root.iterdir()))
        self.assertTrue(all(e['state']=='rejected' and not e['hardware_action'] for e in self.events()))

    def test_missing_session_invalid_root_unregistered_and_no_upward_search(self):
        env=self.env.copy(); env.pop('PTOBENCH_SESSION_ROOT')
        process,result=self.cli(['evaluate'],env=env)
        self.assertEqual((process.returncode,result['error_code']),(2,'invalid_request'))
        self.assertEqual(self.events(),[])
        env={**self.env,'PTOBENCH_SESSION_ROOT':str(self.repo/'missing')}
        process,result=self.cli(['evaluate'],env=env)
        self.assertEqual((process.returncode,result['error_code']),(2,'invalid_request'))
        unregistered=self.handle.session_root/'workspaces/unknown'; unregistered.mkdir()
        process,result=self.cli(['evaluate',str(unregistered)])
        self.assertEqual((process.returncode,result['error_code']),(2,'invalid_workspace'))
        sub=self.handle.initial_workspace/'sub'; sub.mkdir()
        process,result=self.cli(['evaluate'],cwd=sub)
        self.assertEqual((process.returncode,result['error_code']),(2,'invalid_workspace'))
        self.assertEqual(len(self.calls()),0)
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],0)

    def test_new_context_public_names_and_old_entry_absence(self):
        metadata=self.handle.initial_workspace/'.benchmark_context'
        modes=common.read_json(metadata/'profiling_modes.json')['modes']
        self.assertEqual(set(modes),{'runtime','pmu','compiler'})
        readme=(metadata/'README.md').read_text()
        self.assertIn('ptobench evaluate .',readme)
        self.assertIn('ptobench profile pmu .',readme)
        self.assertNotIn('bench profile hardware',readme)
        self.assertNotIn('profile compile ',readme)
        self.assertFalse((SOURCE_BENCHMARK/'bench').exists())
        self.assertFalse((SOURCE_BENCHMARK/'evaluator/bench.py').exists())
        for path in ('evaluate.py','prepare_ref.py','generate_fixtures.py','workspace.py'):
            self.assertTrue((SOURCE_BENCHMARK/'evaluator'/path).is_file())

    def test_linked_source_is_rejected_before_reservation(self):
        sentinel=self.repo/'sentinel'; sentinel.write_text('outside')
        helper=self.handle.initial_workspace/'helper.py'
        helper.unlink(); helper.symlink_to(sentinel)
        for args in (['evaluate'],['profile','runtime'],['fork']):
            process,result=self.cli(args)
            self.assertEqual((process.returncode,result['error_code']),(2,'invalid_workspace'))
        self.assertEqual(sentinel.read_text(),'outside')
        self.assertEqual(len(self.calls()),0)
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],0)
        self.assertFalse(list(self.handle.snapshots_root.iterdir()))

    def test_legacy_session_cannot_restore_removed_parser(self):
        old_parser=self.bench/'evaluator/bench.py'
        old_parser.write_text('raise AssertionError("retired parser must never run")\n')
        legacy=self.create(budget=2,optimizer='pi')
        old_parser.unlink()
        env={**self.env,'PTOBENCH_SESSION_ROOT':str(legacy.session_root)}
        process,result=self.cli(['evaluate',str(legacy.initial_workspace)],env=env)
        self.assertEqual((process.returncode,result['error_code']),(5,'infrastructure_error'))
        self.assertIn('retired CLI',result['message'])
        self.assertFalse(old_parser.exists())
        self.assertFalse((self.bench/'bench').exists())
        self.assertEqual(len(self.calls()),0)
        self.assertEqual(common.read_json(legacy.session_root/'session.json')['usage']['hardware_actions'],0)
        self.assertFalse(list(legacy.snapshots_root.iterdir()))
