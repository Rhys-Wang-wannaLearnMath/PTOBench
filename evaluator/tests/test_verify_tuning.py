"""Evidence validation must not turn failed/diagnostic runs into hardware PASS."""
import copy
from pathlib import Path
import sys
import unittest
import tempfile
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common
import verify_tuning


class EvidenceTests(unittest.TestCase):
    def test_baseline_reuse_accepts_complete_measurements_without_optional_tuning_metadata(self):
        case = {'case_id': 'test_case', 'fixture_ids': ['fixture_000']}
        defaults = common.tuning_defaults(case)
        baseline = {
            'status': 'ready',
            'compiler_tuning': defaults,
            'fixtures': {'fixture_000': {'samples_effective_us': [1., 2., 3.]}},
        }
        self.assertTrue(verify_tuning.baseline_measurement_complete(baseline, case))
        baseline['tuning_identity'] = {'source_sha256': 'old'}
        baseline['tuning_evidence'] = None
        baseline['resolved_tuning'] = None
        self.assertTrue(verify_tuning.baseline_measurement_complete(baseline, case))
        baseline['fixtures']['fixture_000']['samples_effective_us'][0] = 0
        self.assertFalse(verify_tuning.baseline_measurement_complete(baseline, case))

    def test_complete_fixture_coverage_cannot_drop_a_failing_second_input(self):
        request = dict(common.TUNING_DEFAULTS)
        case = {'fixture_ids':['fixture_000','fixture_001']}
        result = {'profiling_mode':'clean','verdict':'PASS','effective_tuning':request,
                  'runs':[{'fixture_id':f, 'passed':True, 'samples_effective_us':[1.,2.,3.]}
                          for f in case['fixture_ids']]}
        self.assertTrue(verify_tuning.valid_result(result, request, case))
        for corruption in ('missing','duplicate','failed'):
            bad = copy.deepcopy(result)
            if corruption == 'missing':
                bad['runs'].pop()
            elif corruption == 'duplicate':
                bad['runs'][1]['fixture_id'] = 'fixture_000'
            else:
                bad['runs'][1]['passed'] = False
            self.assertFalse(verify_tuning.valid_result(bad, request, case))

    def test_default_baseline_promotion_requires_identical_ref_and_full_formal_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir, sealed = root/'case', root/'sealed'
            (case_dir/'seed').mkdir(parents=True)
            (sealed/'ref').mkdir(parents=True)
            (case_dir/'seed/candidate.py').write_text('def entry(): pass\n')
            (sealed/'ref/ref_npu.py').write_text('def entry(): pass\n')
            old={'status':'ready','compiler_tuning':dict(common.TUNING_DEFAULTS)}
            common.write_json(sealed/'ref_performance.json',old)
            request={**common.TUNING_DEFAULTS,'ring_task_window':16384}
            row={'tuning':request,'evidence':'job','identity':{'source_sha256':'test'}}
            case={'workspace':'seed','module':'candidate.py','fixture_ids':['fixture_000']}
            common.write_json(sealed/'certification.json', {'schema_version':1, 'status':'ready'})
            run={'passed':True,'fixture_id':'fixture_000', 'samples_effective_us':[1.,2.,3.],
                 'raw_samples_effective_us':[1.,2.,3.], 'median_effective_us':2.,'mean_effective_us':2.,
                 'measurement_config':{'actual_rounds':3,'actual_warmup':2,'tuning':{'requested':request}}}
            result={'profiling_mode':'clean','phase':'all','verdict':'PASS','effective_tuning':request,'runs':[run]}
            with mock.patch.object(common,'BENCHMARK_ROOT',root), \
                 mock.patch.object(common,'tuning_defaults',return_value=request), \
                 mock.patch.object(common,'benchmark_sizes',return_value=(2,3)):
                bad=copy.deepcopy(result);bad['runs'][0]['measurement_config']['actual_rounds']=2
                with self.assertRaises(ValueError):
                    verify_tuning.publish_default_baseline(row,case_dir,sealed,case,bad)
                self.assertEqual(common.read_json(sealed/'ref_performance.json'),old)
                (sealed/'ref/ref_npu.py').write_text('def entry(): return 1\n')
                with self.assertRaises(ValueError):
                    verify_tuning.publish_default_baseline(row,case_dir,sealed,case,result)
                (sealed/'ref/ref_npu.py').write_text('def entry(): pass\n')
                verify_tuning.publish_default_baseline(row,case_dir,sealed,case,result)
                self.assertEqual(common.read_json(root/'job/previous_ref_performance.json'),old)
                self.assertEqual(common.read_json(sealed/'ref_performance.json')['compiler_tuning'],request)
                self.assertTrue(row['baseline_published'])

    def test_acceptance_schedules_defaults_only(self):
        self.assertEqual(verify_tuning.requests('defaults'), [common.TUNING_DEFAULTS])
        combinations = verify_tuning.requests('combinations')
        self.assertEqual(len(combinations), 6)
        self.assertIn(common.TUNING_DEFAULTS, combinations)
        with self.assertRaises(ValueError):
            verify_tuning.requests('unknown')

    def test_requires_correctness_clean_matching_configuration_and_positive_samples(self):
        request = dict(common.TUNING_DEFAULTS)
        result = {'profiling_mode':'clean','verdict':'PASS','effective_tuning':request,
                  'runs':[{'passed':True,'samples_effective_us':[1.,2.,3.]}]}
        self.assertTrue(verify_tuning.valid_result(result,request))
        for field,value in (('profiling_mode','runtime'),('verdict','FAIL'),('effective_tuning',{}),('runs',[])):
            modified=copy.deepcopy(result);modified[field]=value
            self.assertFalse(verify_tuning.valid_result(modified,request))
        for samples in ([],[0.,1.,2.],[float('nan'),1.,2.],[float('inf'),1.,2.],['1',2.,3.]):
            modified=copy.deepcopy(result);modified['runs'][0]['samples_effective_us']=samples
            self.assertFalse(verify_tuning.valid_result(modified,request))
        modified=copy.deepcopy(result);modified['runs'][0]['passed']=False
        self.assertFalse(verify_tuning.valid_result(modified,request))


if __name__=='__main__':unittest.main()
