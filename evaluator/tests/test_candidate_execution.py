"""Output projection and auxiliary invariants survive Candidate isolation."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_execution as execution
import common
from golden import ScalarSpec, TensorSpec
import golden.runner as runner


class ValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = self.root / 'fixture'
        (self.fixture / 'out').mkdir(parents=True)
        official = common.load_module(
            common.BENCHMARK_ROOT / 'sealed_assets/deepseek_v4/'
            'deepseek_v4_decode_compressor_ratio0/evaluate.py', 'ratio0_validation_test')
        self.compare = {'out': lambda actual, expected, **_: (
            bool(torch.allclose(actual, expected, rtol=.001, atol=.001)), 'output mismatch'),
            **official._invariant_compare()}
        self.values = {'out': torch.zeros(2), 'kv_state': torch.ones(2),
                       'score_state': torch.ones(2)}
        self.specs = []
        for name, value in self.values.items():
            torch.save(value, self.fixture / 'out' / (name + '.pt'))
            spec = TensorSpec(name, [2], torch.float32)
            spec.direction = 'out' if name == 'out' else 'inout'
            self.specs.append(spec)
        records = [execution.spec_record(s) for s in self.specs]
        self.validation = execution.Validation(self.root, {'fixture_000': (self.fixture, records)},
                                               list(self.compare), self.compare)
        self.addCleanup(self.validation.close)
        environment = patch.dict(os.environ, {
            'PTOBENCH_WORKER_WIRE': str(self.validation.wire),
            'PTOBENCH_VALIDATION_SOCKET': str(self.validation.socket_path),
            'PTOBENCH_SKIP_CORRECTNESS': '0'})
        environment.start()
        self.addCleanup(environment.stop)

    def replay(self, specs=None, values=None):
        with execution.bridge('fixture_000'):
            # A permissive worker comparator cannot bypass the host comparator.
            runner._validate(specs or self.specs, values or self.values, {}, 1, 1,
                             {name: lambda *_args, **_kw: (True, '') for name in self.values})

    def test_output_and_both_unchanged_states_pass(self):
        self.replay()
        self.assertEqual(self.validation.validated, {'fixture_000'})
        self.assertFalse(self.validation.failed)

    def test_worker_bridge_supplies_typed_fixture_scalar_to_host_comparator(self):
        (self.fixture / 'in').mkdir()
        torch.save(torch.tensor(128, dtype=torch.int32), self.fixture / 'in/num_tokens.pt')
        self.validation.fixtures['fixture_000'][1].append(
            execution.spec_record(ScalarSpec('num_tokens', torch.int32, 1)))
        observed = []
        def compare(actual, expected, *, inputs, **_):
            scalar = inputs['num_tokens']
            observed.append(scalar)
            return (isinstance(scalar, torch.Tensor) and scalar.ndim == 0
                    and scalar.dtype == torch.int32 and scalar.item() == 128
                    and torch.equal(actual, expected)), 'typed scalar or output mismatch'
        self.compare['out'] = compare
        self.replay()
        self.assertEqual(len(observed), 1)
        self.assertFalse(self.validation.failed)

    def test_either_state_change_below_allclose_tolerance_is_rejected(self):
        for name in ('kv_state', 'score_state'):
            with self.subTest(name=name):
                values = {key: value.clone() for key, value in self.values.items()}
                values[name][0] += .0001
                with self.assertRaisesRegex(AssertionError, 'state exact'):
                    self.replay(values=values)
        self.assertTrue(self.validation.failed)

    def test_wrong_public_output_is_rejected(self):
        values = dict(self.values, out=torch.ones(2))
        with self.assertRaisesRegex(AssertionError, 'output mismatch'):
            self.replay(values=values)

    def test_missing_or_unknown_outputs_are_rejected(self):
        extra = TensorSpec('unknown', [2], torch.float32)
        extra.direction = 'out'
        for specs in (self.specs[:1], self.specs + [extra]):
            with self.subTest(names=[s.name for s in specs]):
                with self.assertRaisesRegex(AssertionError, 'output names disagree'):
                    self.replay(specs=specs, values={**self.values, 'unknown': torch.ones(2)})

    def test_profile_skip_does_not_install_validation_hooks(self):
        originals = runner._compute_golden, runner._validate
        with patch.dict(os.environ, {'PTOBENCH_SKIP_CORRECTNESS': '1'}):
            with execution.bridge('fixture_000'):
                self.assertEqual((runner._compute_golden, runner._validate), originals)

    def test_close_removes_successful_transport_and_preserves_records(self):
        self.replay()
        record = self.root / 'authority_result.json'
        record.write_text('formal result')
        self.assertTrue(list(self.validation.wire.glob('*.bin')))
        self.validation.close()
        self.assertFalse(self.validation.wire.exists())
        self.assertEqual(record.read_text(), 'formal result')
        self.assertTrue(torch.equal(torch.load(self.fixture / 'out/out.pt', weights_only=True),
                                    self.values['out']))
        self.validation.close()  # Normal cleanup is idempotent.

    def test_close_removes_transport_after_comparison_failure(self):
        with self.assertRaisesRegex(AssertionError, 'output mismatch'):
            self.replay(values=dict(self.values, out=torch.ones(2)))
        self.validation.close()
        self.assertFalse(self.validation.wire.exists())
        self.assertTrue(self.validation.failed)

    def test_transport_cleanup_does_not_follow_worker_symlinks(self):
        outside = self.root / 'retained'
        outside.mkdir()
        sentinel = outside / 'golden.pt'
        sentinel.write_bytes(b'RETAINED')
        (self.validation.wire / 'escape').symlink_to(outside, target_is_directory=True)
        self.validation.close()
        self.assertFalse(self.validation.wire.exists())
        self.assertEqual(sentinel.read_bytes(), b'RETAINED')


class PreparationCleanupTests(unittest.TestCase):
    def test_setup_error_closes_validation_and_removes_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = root / 'sealed'
            fixture = sealed / 'fixtures/fixture_000'
            (fixture / 'out').mkdir(parents=True)
            torch.save(torch.zeros(2), fixture / 'out/out.pt')
            for name in ('certification.json', 'ref_performance.json'):
                (sealed / name).write_text('{}')
            case_dir = root / 'case'
            (case_dir / 'public').mkdir(parents=True)
            (case_dir / 'public/comparator.py').write_text('# public comparator\n')
            candidate = root / 'action/workspace'
            candidate.mkdir(parents=True)
            case = {'fixture_ids': ['fixture_000'], 'correctness': {'outputs': ['out']},
                    'comparator': 'public/comparator.py'}
            arguments = {'case': case, 'sealed': sealed, 'case_dir': case_dir,
                         'candidate_workspace': candidate, 'platform': 'cpu',
                         'build_specs': lambda _: [TensorSpec('out', [2], torch.float32)],
                         'make_compare': lambda: {'out': lambda *_a, **_k: (True, '')}}
            with patch.object(execution.common, 'read_tuning', return_value={}), \
                 patch.object(execution.common, 'environment', side_effect=RuntimeError('setup failed')):
                with self.assertRaisesRegex(RuntimeError, 'setup failed'):
                    execution.isolated_evaluate(arguments)
            self.assertFalse((candidate.parent / 'isolated_worker/wire').exists())
            self.assertTrue((fixture / 'out/out.pt').is_file())


class WorkerDiagnosticsTests(unittest.TestCase):
    def test_worker_logs_and_lifecycle_are_retained_on_missing_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            action = root / 'action'
            candidate = action / 'workspace'
            result = action / 'isolated_worker/result'
            candidate.mkdir(parents=True)
            result.mkdir(parents=True)
            lifecycle = {'schema_version': 1, 'exception_type': 'RuntimeError',
                         'message': 'compiler exited', 'events': [
                             {'stage': 'compile', 'event_type': 'stage', 'status': 'started'}]}
            common.write_json(result / 'execution.json', lifecycle)
            host_lifecycle = action / 'execution.json'
            with patch.dict(os.environ, {'PTOBENCH_EXECUTION_RECORD': str(host_lifecycle)}):
                logs = execution._worker_diagnostics_directory(candidate, 'fixture_000', result)
                (logs / 'worker_stdout.log').write_bytes(b'compile output')
                (logs / 'worker_stderr.log').write_bytes(b'compile error')
                process, worker_execution = execution._retain_worker_diagnostics(
                    'fixture_000', 1, result, logs)
            self.assertEqual((logs / 'worker_stdout.log').read_bytes(), b'compile output')
            self.assertEqual((logs / 'worker_stderr.log').read_bytes(), b'compile error')
            self.assertEqual(common.read_json(host_lifecycle), lifecycle)
            self.assertEqual(worker_execution, lifecycle)
            self.assertEqual(process, {'schema_version': 1, 'exit_code': 1,
                                       'summary_present': False, 'execution_status': 'available'})



class SpecProjectionTests(unittest.TestCase):
    def test_all_six_official_output_projections_preserve_inout_reset_membership(self):
        for case_path in sorted((common.BENCHMARK_ROOT / 'sealed_assets/deepseek_v4_flash_dspark').glob('prefill_[ch]ca_*/evaluate.py')):
            with self.subTest(case=case_path.parent.name):
                official = common.load_module(case_path, 'projection_test_' + case_path.parent.name)
                cls = getattr(official, '_CorrectnessTensorSpec', None) or official._CorrectnessSpec
                specs = [cls(name, [2], torch.float32) for name in ('x_out', 'kv_cache', 'compress_state')]
                with tempfile.TemporaryDirectory() as temporary:
                    rebuilt = execution.worker_specs([execution.spec_record(s) for s in specs], Path(temporary))
                for source, target in zip(specs, rebuilt):
                    for direction in ('in', 'out', 'inout'):
                        source.direction = target.direction = direction
                        self.assertEqual(target.is_output, source.is_output)
                        self.assertEqual(target.is_input, source.is_input)
                self.assertTrue(rebuilt[-1].is_input)
                self.assertFalse(rebuilt[-1].is_output)


class InputDictTests(unittest.TestCase):
    def test_dict_type_and_metadata_access_remain_lazy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            (fixture / 'in').mkdir()
            expected = torch.tensor([2, 3], dtype=torch.int32)
            torch.save(expected, fixture / 'in/chunk_lens.pt')
            records = [execution.spec_record(TensorSpec(name, [2], torch.int32))
                       for name in ('chunk_lens', 'unread_weight', 'out')]
            with patch.object(execution.torch, 'load', wraps=torch.load) as load:
                inputs = execution.Inputs(records, fixture, outputs=['out'])
                self.assertIsInstance(inputs, dict)
                self.assertEqual(load.call_count, 0)
                self.assertEqual(set(inputs.keys()), {'chunk_lens', 'unread_weight'})
                self.assertEqual(len(inputs), 2)
                self.assertTrue(inputs)
                self.assertTrue(torch.equal(inputs.get('chunk_lens'), expected))
                self.assertIs(inputs['chunk_lens'], inputs.get('chunk_lens'))
                self.assertEqual(load.call_count, 1)
                marker = object()
                self.assertIs(inputs.get('missing', marker), marker)
                with self.assertRaises(KeyError):
                    inputs['out']

    def test_dict_views_conversions_and_copies_return_loaded_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            (fixture / 'in').mkdir()
            records = []
            for name, value in (('a', 2), ('b', 3)):
                torch.save(torch.tensor(value, dtype=torch.int32), fixture / 'in' / (name + '.pt'))
                records.append(execution.spec_record(ScalarSpec(name, torch.int32, value)))
            operations = (dict, lambda x: x.copy(), lambda x: dict(x.items()),
                          lambda x: {**x}, lambda x: dict(**x))
            for operation in operations:
                inputs = execution.Inputs(records, fixture)
                result = operation(inputs)
                self.assertEqual({k: v.item() for k, v in result.items()}, {'a': 2, 'b': 3})
            inputs = execution.Inputs(records, fixture)
            self.assertEqual([v.item() for v in inputs.values()], [2, 3])

    def test_official_chunk_comparator_keeps_metadata_and_numerical_checks(self):
        case_dir, sealed, case = common.case_paths('deepseek_v4.prefill_layer_ep4_hca_flash')
        official = common.load_module(sealed / 'evaluate.py', 'chunk_dict_validation_test')
        compare = official._compare(case_dir, case)['x_next']
        fixture = sealed / 'fixtures/fixture_000'
        records = []
        for name in ('chunk_lens', 'chunk_offsets'):
            value = torch.load(fixture / 'in' / (name + '.pt'), map_location='cpu', weights_only=True)
            records.append(execution.spec_record(TensorSpec(name, list(value.shape), value.dtype)))
        # This inaccessible unused weight must never be loaded by comparison.
        records.append(execution.spec_record(TensorSpec('unread_weight', [1], torch.float32)))
        inputs = execution.Inputs(records, fixture)
        expected = torch.zeros_like(torch.load(fixture / 'out/x_next.pt', map_location='cpu', weights_only=True))
        self.assertTrue(compare(expected, expected, inputs=inputs)[0])
        wrong = torch.full_like(expected, 100)
        self.assertFalse(compare(wrong, expected, inputs=inputs)[0])
        metadata = {name: inputs[name] for name in ('chunk_lens', 'chunk_offsets')}
        bad = {**metadata, 'chunk_lens': metadata['chunk_lens'].to(torch.int64)}
        self.assertIn('invalid chunk metadata', compare(expected, expected, inputs=bad)[1])
        bad = {**metadata, 'chunk_lens': metadata['chunk_lens'].clone()}
        bad['chunk_lens'][0, 0] += 1
        self.assertIn('rank metadata mismatch', compare(expected, expected, inputs=bad)[1])


class ScalarInputTests(unittest.TestCase):
    def test_host_inputs_match_effective_golden_scalars_for_all_supported_dtypes(self):
        from golden.spec import SUPPORTED_SCALAR_DTYPES
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            (fixture / 'in').mkdir()
            for scalar_dtype in SUPPORTED_SCALAR_DTYPES:
                with self.subTest(dtype=scalar_dtype):
                    spec = ScalarSpec('n', scalar_dtype, torch.tensor(1, dtype=scalar_dtype),
                                      compile_runtime=True)
                    record = execution.spec_record(spec)
                    # Fixture replay is authoritative, including when its value
                    # differs from the spec's construction-time value.
                    cached = torch.tensor(0, dtype=scalar_dtype)
                    torch.save(cached, fixture / 'in/n.pt')
                    expected = runner._effective_scalar_specs([spec], fixture)['n'].value
                    actual = execution.Inputs([record], fixture)['n']
                    self.assertIsInstance(actual, torch.Tensor)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertTrue(torch.equal(actual, expected))

    def test_scalar_cache_with_wrong_dtype_or_shape_is_rejected(self):
        record = execution.spec_record(ScalarSpec('n', torch.int32, 128))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            (fixture / 'in').mkdir()
            for invalid in (torch.tensor(128, dtype=torch.int64),
                            torch.tensor([128], dtype=torch.int32), 128):
                with self.subTest(value=invalid):
                    torch.save(invalid, fixture / 'in/n.pt')
                    with self.assertRaisesRegex(ValueError, 'scalar ABI mismatch'):
                        execution.Inputs([record], fixture)['n']

    def test_all_nine_prefill_comparators_receive_typed_scalar_and_still_reject_bad_outputs(self):
        for kind in ('swa', 'hca', 'csa'):
            for variant in ('demo', 'flash', 'pro'):
                case_id = f'deepseek_v4_flash_dspark.prefill_{kind}_{variant}'
                with self.subTest(case=case_id):
                    case_dir, sealed, case = common.case_paths(case_id)
                    official = common.load_module(sealed / 'evaluate.py', 'scalar_test_' + kind + variant)
                    compare = official.make_compare(case_dir, case)['x_out']
                    fixture = sealed / 'fixtures/fixture_000'
                    scalar = torch.load(fixture / 'in/num_tokens.pt', map_location='cpu', weights_only=True)
                    record = execution.spec_record(ScalarSpec('num_tokens', torch.int32, scalar))
                    inputs = execution.Inputs([record], fixture)
                    expected = torch.zeros_like(torch.load(fixture / 'out/x_out.pt', map_location='cpu', weights_only=True))
                    self.assertTrue(compare(expected.clone(), expected, inputs=inputs)[0])
                    actual = expected.clone()
                    actual[:int(scalar.item())] = 100
                    self.assertFalse(compare(actual, expected, inputs=inputs)[0])


if __name__ == '__main__':
    unittest.main()
