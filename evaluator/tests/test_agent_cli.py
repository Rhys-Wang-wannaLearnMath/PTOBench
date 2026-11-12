from pathlib import Path
import json
import os
import stat
import subprocess
import sys

from action_test_support import ActionTestBase, SOURCE_BENCHMARK
import common
import session


class AgentCliTest(ActionTestBase):
    def test_forked_workers_are_real_roots_with_isolated_readonly_outputs(self):
        def assert_real_tree(root):
            for name in ('candidate', '.benchmark_context', '.benchmark_outputs'):
                self.assertTrue(stat.S_ISDIR((root / name).lstat().st_mode), root / name)
                self.assertFalse((root / name).is_symlink(), root / name)

        initial = self.handle.initial_workspace
        assert_real_tree(initial)
        process, first = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 0, first)
        first_output = initial / '.benchmark_outputs/clean/latest_session_result.json'
        self.assertTrue(first_output.is_file())

        process, fork = self.cli(['fork'])
        self.assertEqual(process.returncode, 0, fork)
        child = Path(fork['workspace'])
        assert_real_tree(child)
        self.assertEqual((child / 'candidate/helper.py').read_text(), 'VALUE = 1\n')
        self.assertNotEqual((child / 'candidate').stat().st_ino, (initial / 'candidate').stat().st_ino)

        (child / 'candidate/helper.py').write_text('VALUE = 2\n')
        self.assertEqual((initial / 'candidate/helper.py').read_text(), 'VALUE = 1\n')
        process, second = self.cli(['evaluate', str(child)])
        self.assertEqual(process.returncode, 0, second)
        second_output = child / '.benchmark_outputs/clean/latest_session_result.json'
        self.assertTrue(second_output.is_file())
        self.assertEqual(json.loads(first_output.read_text())['event_seq'], first['event_seq'])
        self.assertEqual(json.loads(second_output.read_text())['event_seq'], second['event_seq'])
        self.assertNotEqual(first['event_seq'], second['event_seq'])

        import grp
        share_gid = grp.getgrnam('ptobench-share').gr_gid
        for worker in (initial, child):
            output = worker / '.benchmark_outputs'
            self.assertEqual(output.stat().st_gid, share_gid)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o750)
            self.assertEqual((output / 'clean/latest_session_result.json').stat().st_gid, share_gid)
            self.assertEqual(stat.S_IMODE((worker / '.benchmark_outputs/clean').stat().st_mode), 0o750)
            self.assertEqual(stat.S_IMODE((worker / '.benchmark_outputs/clean/latest_session_result.json').stat().st_mode), 0o640)
            for base, directories, files in os.walk(output):
                for name in directories:
                    entry = Path(base) / name
                    self.assertEqual(entry.stat().st_gid, share_gid)
                    self.assertEqual(stat.S_IMODE(entry.stat().st_mode), 0o750)
                for name in files:
                    entry = Path(base) / name
                    self.assertEqual(entry.stat().st_uid, os.getuid())
                    self.assertEqual(entry.stat().st_gid, share_gid)
                    self.assertEqual(stat.S_IMODE(entry.stat().st_mode), 0o640)

    def test_legacy_session_cannot_restore_the_unrepaired_environment(self):
        marker = self.bench / 'evaluator/runtime_policy.py'
        content = marker.read_bytes()
        marker.unlink()
        try:
            old = self.create()
        finally:
            marker.write_bytes(content)
        environment = {**self.env, 'PTOBENCH_SESSION_ROOT': str(old.session_root)}
        process, result = self.cli(['evaluate'], cwd=old.initial_workspace, env=environment)
        self.assertEqual(process.returncode, 5, result)
        self.assertIn('fresh Session', result['message'])
        self.assertEqual(marker.read_bytes(), content)
        self.assertEqual(self.calls(), [])

    def test_help_without_session_exposes_profiles_without_reserving_budget(self):
        environment = dict(self.env)
        environment.pop('PTOBENCH_SESSION_ROOT')
        for args in (['--help'], ['profile', '--help']):
            with self.subTest(args=args):
                process = subprocess.run([self.python, str(self.bench / 'ptobench'), *args],
                                         env=environment, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, timeout=10)
                self.assertEqual(process.returncode, 0, process.stderr.decode())
                self.assertIn('profile', process.stdout.decode())
                if args[0] == 'profile':
                    help_text = ' '.join(process.stdout.decode().split())
                    self.assertIn('Profile does not check output correctness', help_text)
                    self.assertIn('ptobench evaluate', help_text)
                    self.assertIn('combined', process.stdout.decode())
        self.assertEqual(self.events(), [])
        self.assertEqual(self.calls(), [])
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'], 0)

    def test_five_operations_in_explicit_split_session(self):
        self.handle = self.create(budget=5, profile_strategy='split')
        self.env['PTOBENCH_SESSION_ROOT'] = str(self.handle.session_root)
        for args, phase, internal in ((['evaluate'],'all','clean'), (['profile','runtime'],'correctness','runtime'),
                                    (['profile','pmu'],'correctness','hardware'), (['profile','compiler'],'correctness','compile')):
            with self.subTest(args=args):
                process, result = self.cli(args)
                self.assertEqual(process.returncode, 0, process.stderr.decode())
                self.assertEqual(result['status'], 'PASS', result)
                call=self.calls()[-1]
                self.assertEqual((call['phase'],call['profile_mode']), (phase,internal))
                self.assertEqual(call['measurement_mode'], 'exploration' if args[0]=='evaluate' else 'formal')
                self.assertEqual(call['case'],self.case_id)
                self.assertNotEqual(call['candidate'],str(self.handle.initial_workspace))
                self.assertEqual(result['latency_us'], 10.0 if internal=='clean' else None)
                self.assertEqual(call['skip_correctness'], args[0] == 'profile')
                if internal == 'clean':
                    self.assertEqual(result['correctness'], 'PASS')
                else:
                    self.assertNotIn('correctness', result)
                if internal == 'clean':
                    self.assertTrue((self.snapshot(result)/'source/candidate/helper.py').is_file())
                else:
                    self.assertIsNone(result['snapshot_id'])
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
        self.handle = self.create(profile_strategy='split')
        self.env['PTOBENCH_SESSION_ROOT'] = str(self.handle.session_root)
        requests = [['profile','hardware'],['profile','compile'],['evaluate','--phase','all'],
                    ['evaluate','--candidate','.'],['evaluate','--result','result.json'],['evaluate','--workspace','.'],
                    ['evaluate','--measurement-mode','formal'],['evaluate','--warmup','1'],['evaluate','--rounds','1'],
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

    def test_new_context_defaults_to_combined(self):
        metadata=self.handle.initial_workspace/'.benchmark_context'
        self.assertFalse((metadata/'profiling_modes.json').exists())
        self.assertFalse((metadata/'fixture_catalog.json').exists())
        readme=(metadata/'README.md').read_text()
        self.assertIn('ptobench evaluate .',readme)
        self.assertIn('ptobench profile .',readme)
        self.assertNotIn('ptobench profile runtime',readme)
        self.assertNotIn('bench profile hardware',readme)
        self.assertNotIn('profile compile ',readme)
        self.assertFalse((SOURCE_BENCHMARK/'bench').exists())
        self.assertFalse((SOURCE_BENCHMARK/'evaluator/bench.py').exists())
        for path in ('evaluate.py','prepare_ref.py','generate_fixtures.py','workspace.py'):
            self.assertTrue((SOURCE_BENCHMARK/'evaluator'/path).is_file())

    def test_combined_profile_matches_evaluate_workspace_syntax(self):
        metadata = self.handle.initial_workspace / '.benchmark_context'
        self.assertFalse((metadata / 'profiling_modes.json').exists())
        self.assertFalse((metadata / 'fixture_catalog.json').exists())
        readme = (metadata / 'README.md').read_text()
        self.assertIn('`ptobench profile .`', readme)
        self.assertNotIn('`ptobench profile runtime', readme)

        process, result = self.cli(['profile'])
        self.assertEqual(process.returncode, 0, result)
        call = self.calls()[-1]
        self.assertEqual((call['profile_mode'], call['phase']), ('combined', 'correctness'))
        self.assertTrue(call['skip_correctness'])
        process, rejected = self.cli(['profile', 'runtime'])
        self.assertEqual(process.returncode, 2, rejected)
        self.assertEqual(rejected['error_code'], 'invalid_workspace')

    def test_linked_source_is_rejected_before_reservation(self):
        sentinel=self.repo/'sentinel'; sentinel.write_text('outside')
        helper=self.handle.initial_workspace/'candidate/helper.py'
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
        self.assertEqual((process.returncode,result['error_code']),(2,'invalid_request'))
        self.assertIn('fresh Session',result['message'])
        self.assertFalse(old_parser.exists())
        self.assertFalse((self.bench/'bench').exists())
        self.assertEqual(len(self.calls()),0)
        self.assertEqual(common.read_json(legacy.session_root/'session.json')['usage']['hardware_actions'],0)
        self.assertFalse(list(legacy.snapshots_root.iterdir()))
