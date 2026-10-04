"""Direct S6 controls share locking, process lifecycle and crash recovery."""
import json
import os
import signal
import subprocess
from unittest import mock

from action_test_support import ActionTestBase
import common
import integrity
import session
import session_actions as actions
import workspace


class MaintenanceExecutionTests(ActionTestBase):
    def prepare(self):
        directory=self.bench/'runs/control'
        directory.mkdir(parents=True)
        workspace.copy_candidate_tree(self.handle.initial_workspace,directory/'workspace')
        argv=[self.python,str(self.bench/'evaluator/evaluate.py'),'--case',self.case_id,
              '--candidate',str(directory/'workspace'),'--phase','all','--profile-mode','clean',
              '--result',str(directory/'result.json')]
        return directory,argv

    def test_requires_lock_and_never_charges_session(self):
        directory,argv=self.prepare()
        with self.assertRaises(integrity.InfrastructureError):
            actions.run_maintenance_process(argv,self.env,directory)
        self.assertFalse((directory/'command.json').exists())
        with session.execution_lock():
            self.assertEqual(actions.run_maintenance_process(argv,self.env,directory),0)
            with self.assertRaises(integrity.InfrastructureError):
                actions.run_maintenance_process(argv,self.env,directory)
        self.assertIn('fake evaluator stdout',(directory/'stdout.log').read_text())
        self.assertIn('fake evaluator stderr',(directory/'stderr.log').read_text())
        self.assertEqual(session.read_session(self.handle.session_root)['usage'],
                         {'hardware_actions':0,'finish_actions':0})
        self.assertFalse(actions._marker_path().exists())

    def start_control(self,directory,argv):
        code=('import sys;sys.path.insert(0,sys.argv[1]);'
              'import session,session_actions,os;'
              '\nwith session.execution_lock(): session_actions.run_maintenance_process('
              'sys.argv[3:],dict(os.environ),sys.argv[2])')
        process=subprocess.Popen([self.python,'-c',code,str(self.bench/'evaluator'),str(directory),*argv],
                                 env=self.env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        return process

    def test_killed_control_owner_is_recovered_before_managed_action(self):
        directory,argv=self.prepare()
        ready=self.repo/'ready'; child=self.repo/'child.pid'
        self.configure(hang=True,ready=str(ready),child=True,child_pid=str(child))
        process=self.start_control(directory,argv)
        self.wait_file(ready,process)
        identity=json.loads((directory/'process_identity.json').read_text())
        process.kill(); process.wait(timeout=5)
        self.assertIsNotNone(session.process_identity(identity['pid']))
        self.configure()
        result,value=self.cli(['evaluate'])
        self.assertEqual(result.returncode,0,value)
        self.assertIsNone(session.process_identity(identity['pid']))
        self.assertIsNone(session.process_identity(int(child.read_text())))
        self.assertTrue(json.loads((directory/'process_result.json').read_text())['recovered'])
        self.assertFalse(actions._marker_path().exists())
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],1)

    def test_signal_stops_control_descendants_and_clears_marker(self):
        directory,argv=self.prepare()
        ready=self.repo/'ready'; child=self.repo/'child.pid'
        self.configure(hang=True,ready=str(ready),child=True,child_pid=str(child))
        process=self.start_control(directory,argv)
        self.wait_file(ready,process)
        identity=json.loads((directory/'process_identity.json').read_text())
        process.send_signal(signal.SIGTERM); process.wait(timeout=20)
        self.assertNotEqual(process.returncode,0)
        self.assertIsNone(session.process_identity(identity['pid']))
        self.assertIsNone(session.process_identity(int(child.read_text())))
        self.assertEqual(json.loads((directory/'process_result.json').read_text())['exit_code'],130)
        self.assertFalse(actions._marker_path().exists())

    def test_unattributed_spawn_and_outside_evidence_fail_closed(self):
        directory,argv=self.prepare()
        with session.execution_lock():
            with self.assertRaises(integrity.InfrastructureError):
                actions.run_maintenance_process(argv,self.env,self.handle.session_root/'staging')
            actions._marker({'schema_version':1,'kind':'maintenance','action_dir':str(directory),
                             'owner':session.process_identity(),'process':None,'state':'starting'})
            with self.assertRaises(integrity.InfrastructureError):
                actions.recover_global_action()
            self.assertTrue(actions._marker_path().exists())
            actions._clear_marker()
