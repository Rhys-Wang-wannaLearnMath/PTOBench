"""Complete fixture coverage at Session, result validation and finalization."""
import os
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

import common
import finalize
import session
import session_actions
import workflow
from session_test_support import SessionTestBase, write_json


class MultiFixtureTest(SessionTestBase):
    def test_workspace_context_does_not_publish_fixture_catalog(self):
        handle=self.create()
        context=handle.initial_workspace/'.benchmark_context'
        assert not (context/'fixture_catalog.json').exists()
        assert common.read_json(context/'case.json')['fixture_ids'] == ['fixture_000']
        sealed=common.case_paths(self.case_id)[1]
        assert common.fixture_input_identity(sealed/'fixtures/fixture_000')

    def two(self):
        case_dir, sealed = common.case_paths(self.case_id)[:2]
        case = common.read_json(case_dir/'case.json')
        case['fixture_ids'] = ['fixture_000','fixture_001']
        write_json(case_dir/'case.json', case)
        shutil.copytree(sealed/'fixtures/fixture_000', sealed/'fixtures/fixture_001')
        manifest = common.read_json(sealed/'fixtures/fixture_001/manifest.json')
        manifest.update(fixture_id='fixture_001', seed=1)
        write_json(sealed/'fixtures/fixture_001/manifest.json', manifest)
        # Distinct actual persisted inputs, not a relabeled directory.
        (sealed/'fixtures/fixture_001/in/x.pt').write_bytes(b'different fixed input')
        performance = common.read_json(sealed/'ref_performance.json')
        performance['fixtures']['fixture_001'] = {'median_effective_us':4.}
        write_json(sealed/'ref_performance.json', performance)
        return case_dir, sealed, case

    def result(self, *, bad=None, diagnostic=False):
        case_dir, sealed, _ = self.two()
        frozen = self.bench/'frozen'
        runs = [{'fixture_id':f'fixture_{i:03d}', 'passed':True,
                 'median_effective_us':v, 'speedup':speed,
                 'input_identity_sha256':common.fixture_input_identity(sealed/'fixtures'/f'fixture_{i:03d}'),
                 'samples_effective_us':[v]*3, 'raw_samples_effective_us':[v]*3,
                 'measurement_config':{'unit':'us','metric':'effective','actual_rounds':3}}
                for i,(v,speed) in enumerate(((10.,2.),(40.,.5)))]
        raw = {'schema_version':1,'case_id':self.case_id,'candidate':str(frozen),
               'phase':'correctness' if diagnostic else 'all',
               'profiling_mode':'runtime' if diagnostic else 'clean',
               'official_score':not diagnostic,'diagnostic_only':diagnostic,'verdict':'PASS','runs':runs}
        if bad == 'missing':
            runs.pop()
        if bad == 'duplicate':
            runs[1]['fixture_id'] = 'fixture_000'
        if bad == 'failure':
            runs[1]['passed'] = False
        if bad == 'zero':
            runs[1]['median_effective_us'] = 0
        if bad == 'raw_changed':
            runs[1]['raw_samples_effective_us'] = [1.]*3
        if bad == 'wrong_input':
            runs[1]['input_identity_sha256'] = runs[0]['input_identity_sha256']
        path = self.bench/'raw.json'
        write_json(path, raw)
        record = {'case_id':self.case_id, 'case_dir':str(case_dir), 'sealed_dir':str(sealed),
                  'optimizer':{'backend':'codex'}}
        return session_actions.validate_result(path, record, frozen, 'profile' if diagnostic else 'evaluate',
                                               'runtime' if diagnostic else 'clean', 0)

    def test_session_validates_all_prepared_inputs_without_generating(self):
        self.two()
        handle = self.create()
        context = common.read_json(handle.initial_workspace/'.benchmark_context/case.json')
        assert context['fixture_ids'] == ['fixture_000','fixture_001']

    def test_second_fixture_missing_file_or_baseline_is_rejected(self):
        _, sealed, _ = self.two()
        (sealed/'fixtures/fixture_001/in/x.pt').unlink()
        with pytest.raises(session.SessionError):
            self.create()

    def test_geometric_aggregate_uses_every_fixture(self):
        raw, observation = self.result()
        assert observation['latency_us'] == pytest.approx(20.)
        assert observation['fixture_count'] == 2 and observation['end_to_end'] is False
        assert observation['metric_name'] == 'geometric_mean_fixture_median_effective_us'
        result = finalize._result({'usage':{'hardware_actions':1,'finish_actions':1}}, raw=raw, intact=True)
        assert result.final_verified_latency_us == pytest.approx(20.)
        assert result.final_verified_speedup == pytest.approx(1.)

    def test_diagnostic_batches_do_not_invent_aggregate_timing(self):
        _, observation = self.result(diagnostic=True)
        assert observation['latency_us'] is None
        assert 'correctness' not in observation

    def test_partial_or_bad_second_results_are_rejected(self):
        # Each instance owns its corpus, avoiding changes to a live Session.
        for bad in ('missing','duplicate','failure','zero','raw_changed','wrong_input'):
            with self.subTest(bad=bad):
                if bad != 'missing':
                    shutil.rmtree(common.case_paths(self.case_id)[1]/'fixtures/fixture_001')
                with pytest.raises(session.SessionError):
                    self.result(bad=bad)


@pytest.mark.parametrize('ids', [[], ['fixture_000']*2, ['../fixture_000'], ['fixture_000', 1],
                               [f'fixture_{i:03d}' for i in range(65)]])
def test_invalid_public_fixture_declarations(ids):
    with pytest.raises(ValueError):
        common.declared_fixture_ids({'fixture_ids':ids})


def test_build_roots_restore_on_errors(tmp_path):
    case = {'fixture_ids':['fixture_000','fixture_001']}
    with patch.dict(os.environ, {'PYPTO_PROG_BUILD_DIR':str(tmp_path)}):
        for fixture in case['fixture_ids']:
            with pytest.raises(RuntimeError), workflow.fixture_build_root(case, fixture):
                assert Path(os.environ['PYPTO_PROG_BUILD_DIR']) == tmp_path/fixture
                raise RuntimeError('runtime failure')
            assert os.environ['PYPTO_PROG_BUILD_DIR'] == str(tmp_path)


def test_partitioned_worker_retains_fixture_build_namespace(tmp_path):
    metadata = tmp_path/'worker.json'
    write_json(metadata, {'fixture_partition':'fixture_001'})
    with patch.dict(os.environ, {'PYPTO_PROG_BUILD_DIR':str(tmp_path/'builds'),
                                 'PTOBENCH_WORKER':'1','PTOBENCH_WORKER_METADATA':str(metadata)}):
        with workflow.fixture_build_root({'fixture_ids':['fixture_001']}, 'fixture_001'):
            assert Path(os.environ['PYPTO_PROG_BUILD_DIR']) == tmp_path/'builds/fixture_001'
        with pytest.raises(ValueError, match='partition mismatch'), workflow.fixture_build_root(
                {'fixture_ids':['fixture_000']}, 'fixture_000'):
            pass
