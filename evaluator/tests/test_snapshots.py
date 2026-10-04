from action_test_support import ActionTestBase
import os
from unittest import mock
import common
import integrity
import session
import session_actions
import workspace


class SnapshotTest(ActionTestBase):
    def test_authority_or_snapshot_registration_failure_never_publishes_success(self):
        for fault in ('authority','snapshot'):
            with self.subTest(fault=fault):
                handle=self.create(budget=2,optimizer='pi')
                real_commit=integrity.commit_records
                real_register=integrity.register_roots
                def commit(protection,updates):
                    if fault=='authority' and any(name.startswith('action_result_') for name in updates):
                        raise integrity.InfrastructureError('authority commit fault')
                    return real_commit(protection,updates)
                def register(protection,roots):
                    if fault=='snapshot' and any(name.startswith('snapshot_') for name in roots):
                        raise integrity.InfrastructureError('Snapshot registration fault')
                    return real_register(protection,roots)
                with mock.patch.dict(os.environ,self.env), \
                     mock.patch.object(integrity,'commit_records',side_effect=commit), \
                     mock.patch.object(integrity,'register_roots',side_effect=register):
                    result=session_actions.run_evaluate(handle.session_root,handle.initial_workspace)
                self.assertEqual((result.exit_code,result.error_code),(5,'infrastructure_error'))
                record=session.read_session(handle.session_root)
                self.assertTrue(record['exploration_closed'])
                self.assertEqual(record['status'],'failed')
                self.assertEqual(record['usage']['hardware_actions'],1)
                self.assertIsNone(result.snapshot_id)
                self.assertEqual((handle.session_root/'events.jsonl').read_text(),'')
                self.assertFalse((handle.initial_workspace/'.benchmark_outputs/clean/latest_session_result.json').exists())
                # Recover the abandoned completion before testing another fault.
                with session.execution_lock():
                    session_actions.recover_global_action()
                event=common.read_json(handle.session_root/'events.jsonl')
                self.assertEqual(event['error_code'],'interrupted')
                self.assertIsNone(event['snapshot_id'])

    def test_identical_successful_sources_have_distinct_observations_and_readonly_trees(self):
        results=[]
        for args in (['evaluate'],['profile','pmu'],['evaluate']):
            process,result=self.cli(args)
            self.assertEqual(process.returncode,0,result)
            results.append(result)
        self.assertEqual(len({r['snapshot_id'] for r in results}),3)
        hashes=[]
        for result in results:
            snap=self.snapshot(result)
            metadata=common.read_json(snap/'metadata.json')
            hashes.append(metadata['content_sha256'])
            self.assertEqual(metadata['event_seq'],result['event_seq'])
            self.assertEqual(metadata['session_id'],session.read_session(self.handle.session_root)['session_id'])
            self.assertEqual(metadata['case_id'],self.case_id)
            self.assertEqual(metadata['observation']['latency_us'],result['latency_us'])
            self.assertEqual((snap/'source/helper.py').stat().st_mode & 0o777,0o444)
            self.assertEqual((snap/'source').stat().st_mode & 0o777,0o555)
            self.assertFalse((snap/'source/.benchmark_context').exists())
            self.assertFalse((snap/'source/.benchmark_outputs').exists())
        self.assertEqual(len(set(hashes)),1)
        self.assertTrue(integrity.check_only(self.handle.protection_root).passed)

    def test_snapshot_and_authority_tampering_restored_before_next_action(self):
        _,result=self.cli(['evaluate'])
        snap=self.snapshot(result)
        source=snap/'source/helper.py'; source.chmod(0o600); source.write_text('rogue')
        metadata=snap/'metadata.json'; metadata.chmod(0o600); metadata.write_text('{}')
        authority=self.handle.session_root/'staging/action_000001/authority_result.json'
        authority.write_text('{}')
        process,next_result=self.cli(['profile','compiler'])
        self.assertEqual(process.returncode,0,next_result)
        self.assertEqual(source.read_text(),'VALUE = 1\n')
        self.assertEqual(common.read_json(metadata)['snapshot_id'],result['snapshot_id'])
        self.assertEqual(common.read_json(authority)['verdict'],'PASS')
        self.assertTrue(integrity.check_only(self.handle.protection_root).passed)
