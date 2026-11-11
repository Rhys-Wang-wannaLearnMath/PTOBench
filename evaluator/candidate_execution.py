"""Isolated Candidate execution with expected tensors kept in the trusted host.

The Case's existing execution wrapper and Golden dispatch/benchmark boundaries
are reused. A process-local validation bridge replaces expected-output loading;
the original comparator executes outside the Candidate's filesystem namespace.
"""
from __future__ import annotations

from collections.abc import ItemsView, ValuesView
import contextlib
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading

import torch

import common
import file_tree
import isolation


def safe_json(path):
    with file_tree.open_file(path) as fd:
        if os.fstat(fd).st_size > 16 * 1024 * 1024:
            raise ValueError('oversized worker metadata')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            return json.load(stream)


def _worker_diagnostics_directory(candidate, fixture_id, result_dir):
    host_record = os.environ.get('PTOBENCH_EXECUTION_RECORD')
    destination = ((Path(host_record).parent / 'worker_diagnostics') if host_record
                   else Path(result_dir) / 'diagnostics') / fixture_id
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _open_worker_log(path):
    with file_tree.open_parent(path) as (parent, name):
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             mode=0o600, dir_fd=parent)
    return os.fdopen(descriptor, 'wb')


def _retain_worker_diagnostics(fixture_id, returncode, result_dir, destination):
    """Link worker lifecycle facts to its streamed output files."""
    host_record = os.environ.get('PTOBENCH_EXECUTION_RECORD')
    execution_path = Path(result_dir) / 'execution.json'
    execution = None
    execution_status = 'unavailable'
    if execution_path.exists():
        try:
            execution = safe_json(execution_path)
            common.write_json(destination / 'worker_execution.json', execution)
            execution_status = 'available'
            if host_record:
                common.write_json(Path(host_record), execution)
        except (OSError, ValueError, TypeError):
            execution_status = 'invalid'

    summary_present = (Path(result_dir) / 'summary.json').is_file()
    process_record = {
        'schema_version': 1,
        'exit_code': returncode,
        'summary_present': summary_present,
        'execution_status': execution_status,
    }
    common.write_json(destination / 'worker_process.json', process_record)
    return process_record, execution


def _worker_failure_message(process_record, destination):
    tail = ''
    for name in ('worker_stderr.log', 'worker_stdout.log'):
        path = destination / name
        if not path.is_file() or not path.stat().st_size:
            continue
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 2000))
            tail = stream.read().decode('utf-8', errors='replace').strip()
        if tail:
            break
    message = f"isolated Candidate did not produce a result (exit={process_record['exit_code']})"
    if tail:
        message += f"; worker output tail: {tail}"
    message += '; full worker logs are archived with this attempt'
    return message


def dtype(name):
    allowed = {str(getattr(torch, n)): getattr(torch, n) for n in (
        'bool', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16',
        'float32', 'float64', 'complex64', 'complex128')}
    if name not in allowed:
        raise ValueError('unsupported tensor wire dtype')
    return allowed[name]


def spec_record(spec):
    from golden import ScalarSpec
    result = {'name': spec.name, 'dtype': str(spec.dtype)}
    if isinstance(spec, ScalarSpec):
        result.update(kind='scalar', value=spec.value.item() if isinstance(spec.value, torch.Tensor) else spec.value,
                      compile_runtime=spec.compile_runtime, benchmark_step=spec.benchmark_step)
    else:
        result.update(kind='tensor', shape=list(spec.shape), resident=spec.resident)
        # Preserve official TensorSpec output projections across serialization.
        # Compile still stamps the real direction; input/reset membership stays
        # independent of which writes the official evaluator validates.
        probe = replace(spec)
        directions = []
        for direction in ('out', 'inout'):
            probe.direction = direction
            if probe.is_output:
                directions.append(direction)
        result['validation_directions'] = directions
    return result


def worker_specs(records, fixture):
    from golden import ScalarSpec, TensorSpec
    class ValidationSpec(TensorSpec):
        @property
        def is_output(self):
            return super().is_output and self.direction in self.validation_directions

    result = []
    for record in records:
        if record['kind'] == 'scalar':
            result.append(ScalarSpec(record['name'], dtype(record['dtype']), record['value'],
                                     compile_runtime=record['compile_runtime'],
                                     benchmark_step=record['benchmark_step']))
        else:
            path = fixture / 'in' / (record['name'] + '.pt')
            init = (lambda path=path: torch.load(path, map_location='cpu', weights_only=True)) if path.exists() else None
            spec = ValidationSpec(record['name'], record['shape'], dtype(record['dtype']),
                                  init_value=init, resident=record['resident'])
            spec.validation_directions = record.get('validation_directions', ('out', 'inout'))
            result.append(spec)
    return result


def configure_devices(runtime, metadata):
    devices = metadata.get('devices')
    if not devices:
        return
    library = ctypes.CDLL('libascendcl.so')
    logical = []
    for physical in devices['physical']:
        value = ctypes.c_int32()
        code = library.aclrtGetLogicDevIdByPhyDevId(physical, ctypes.byref(value))
        if code:
            raise RuntimeError(f'cannot resolve sandbox NPU {physical} (ACL {code})')
        logical.append(value.value)
    runtime.device_id = logical[0]
    if runtime.distributed_config is not None:
        runtime.distributed_config = replace(runtime.distributed_config, device_ids=logical)
    runtime._ptobench_device_mapping = devices


class Inputs(dict):
    """Golden-compatible input dict without eagerly loading all model weights."""
    _unloaded = object()

    def __init__(self, records, fixture, outputs=()):
        self.records = {r['name']: r for r in records if r['kind'] == 'scalar' or r['name'] not in outputs}
        self.fixture = fixture
        super().__init__((name, self._unloaded) for name in self.records)

    def __iter__(self):
        # Keep dict conversion on the mapping protocol so it reads real values.
        return super().__iter__()

    def get(self, name, default=None):
        return self[name] if name in self else default

    def items(self):
        return ItemsView(self)

    def values(self):
        return ValuesView(self)

    def copy(self):
        return dict(self)

    def __getitem__(self, name):
        cached = super().__getitem__(name)
        if cached is not self._unloaded:
            return cached
        record = self.records[name]
        value = torch.load(self.fixture / 'in' / (name + '.pt'), map_location='cpu', weights_only=True)
        if record['kind'] == 'scalar':
            # Match Golden's effective ScalarSpec.value: the trusted fixture's
            # typed zero-dimensional tensor, not the JSON transport primitive.
            if not isinstance(value, torch.Tensor) or value.ndim != 0 or value.dtype != dtype(record['dtype']):
                raise ValueError(f'trusted fixture scalar ABI mismatch: {name}')
        super().__setitem__(name, value)
        return value


class Validation:
    def __init__(self, private, fixtures, outputs, compare):
        # evaluate.py can start outside PTOBench with only the selected source
        # checkout on PYTHONPATH. Resolve the bundled trusted helper explicitly.
        if str(common.BENCHMARK_ROOT) not in sys.path:
            sys.path.insert(0, str(common.BENCHMARK_ROOT))
        self.ipc = tempfile.TemporaryDirectory(prefix='ptobench-validate-')
        self.socket_path = Path(self.ipc.name) / 'rpc'
        self.wire = private / 'wire'
        self.wire.mkdir()
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(str(self.socket_path))
        self.server.listen(4)
        self.server.settimeout(.2)
        self.fixtures, self.outputs, self.compare = fixtures, outputs, compare
        self.validated = set()
        self.failed = False
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        from golden.validation import validate_golden
        while not self.stop.is_set():
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            with connection:
                try:
                    raw = connection.makefile('rb').readline(1024 * 1024 + 1)
                    if len(raw) > 1024 * 1024:
                        raise ValueError('oversized validation request')
                    request = json.loads(raw)
                    if set(request) != {'fixture', 'outputs'}:
                        raise ValueError('invalid validation fields')
                    fixture_id = request['fixture']
                    fixture, records = self.fixtures[fixture_id]
                    expected_specs = {r['name']: r for r in records}
                    if set(request['outputs']) != set(self.outputs):
                        raise ValueError('validation output names disagree with Case')
                    actual, expected = {}, {}
                    for name, item in request['outputs'].items():
                        spec = expected_specs[name]
                        if set(item) != {'file', 'shape', 'dtype'} or item['shape'] != spec['shape'] or item['dtype'] != spec['dtype']:
                            raise ValueError('validation output ABI mismatch')
                        leaf = item['file']
                        if not isinstance(leaf, str) or not leaf.isascii() or not leaf.removesuffix('.bin').isdigit() or not leaf.endswith('.bin'):
                            raise ValueError('invalid tensor wire filename')
                        count = 1
                        for n in spec['shape']:
                            count *= n
                        nbytes = count * torch.empty((), dtype=dtype(spec['dtype'])).element_size()
                        with file_tree.open_file(self.wire / leaf) as fd:
                            if os.fstat(fd).st_size != nbytes:
                                raise ValueError('tensor wire size mismatch')
                            with os.fdopen(os.dup(fd), 'rb') as stream:
                                raw_tensor = bytearray(stream.read(nbytes + 1))
                        if len(raw_tensor) != nbytes:
                            raise ValueError('tensor wire changed during read')
                        actual[name] = (torch.frombuffer(raw_tensor, dtype=dtype(spec['dtype'])).reshape(spec['shape'])
                                        if nbytes else torch.empty(spec['shape'], dtype=dtype(spec['dtype'])))
                        expected[name] = torch.load(fixture / 'out' / (name + '.pt'), map_location='cpu', weights_only=True)
                    # Case comparator closures, never Candidate-supplied tolerance.
                    validate_golden(actual, expected, rtol=1e-3, atol=1e-3,
                                    compare_fn=self.compare,
                                    inputs=Inputs(records, fixture, self.outputs))
                    self.validated.add(fixture_id)
                    reply = {'passed': True}
                except Exception as error:
                    self.failed = True
                    self.error = str(error)
                    reply = {'passed': False, 'error': str(error)}
                try:
                    connection.sendall((json.dumps(reply) + '\n').encode())
                except OSError:
                    pass

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('trusted validation did not stop; retaining tensor transport')
        self.server.close()
        self.ipc.cleanup()
        # Only disposable Candidate transport lives here. Comparisons have
        # finished; formal results, artifacts and shared fixtures stay intact.
        # rmtree does not follow symlinks planted inside the writable wire tree.
        if self.wire.exists():
            shutil.rmtree(self.wire)


def isolated_evaluate(arguments):
    case = arguments['case']
    sealed, case_dir, candidate = (Path(arguments[n]) for n in ('sealed', 'case_dir', 'candidate_workspace'))
    if len(case['fixture_ids']) > 1:
        # Device resources remain resident until the worker exits. Finish each
        # fixture process before the next dep-gen child acquires registers.
        runs = []
        lifecycle_path = os.environ.get('PTOBENCH_EXECUTION_RECORD')
        lifecycle = common.read_json(Path(lifecycle_path)) if lifecycle_path else None
        children = {}
        for fixture_id in common.declared_fixture_ids(case):
            part = {**arguments, 'case':{**case, 'fixture_ids':[fixture_id]},
                    '_fixture_partition':fixture_id}
            rows = isolated_evaluate(part)
            if len(rows) != 1 or rows[0].get('fixture_id') != fixture_id:
                raise ValueError('worker fixture partition result mismatch')
            rows[0]['input_identity_sha256'] = common.fixture_input_identity(sealed/'fixtures'/fixture_id)
            runs.extend(rows)
            if lifecycle_path:
                children[fixture_id] = common.read_json(Path(lifecycle_path))
        if lifecycle is not None:
            lifecycle['fixture_executions'] = children
            lifecycle['events'].extend({**event,'fixture_id':fixture_id}
                for fixture_id,child in children.items() for event in child.get('events',[]))
            for key in ('effective_target','tuning'):
                values = [child.get(key) for child in children.values()]
                if values and all(value == values[0] for value in values):
                    lifecycle[key] = values[0]
            common.write_json(Path(lifecycle_path), lifecycle)
        return runs
    # Build the trusted runtime before serializing specs.  The worker receives
    # the resulting resident ABI; it never gets to choose residency itself.
    updates, runtime = common.environment(
        case, arguments['platform'], tuning=common.read_tuning(candidate, case)
    )
    # No Candidate import has occurred in this process.
    fixtures = {}
    for fixture_id in case['fixture_ids']:
        fixture = sealed / 'fixtures' / fixture_id
        specs = arguments['build_specs'](fixture)
        fixtures[fixture_id] = (fixture, [spec_record(s) for s in specs])
        del specs
    compare = arguments['make_compare']()
    if set(compare) != set(case['correctness']['outputs']):
        raise RuntimeError('isolated evaluation requires explicit Case comparators for every output')
    # Only the trusted official evaluator may add auxiliary invariants. Never
    # accept comparator names or callbacks supplied by the Candidate worker.
    make_invariants = arguments.get('make_invariant_compare')
    invariants = make_invariants() if make_invariants else {}
    if set(invariants) & set(compare):
        raise RuntimeError('auxiliary invariants must not override public comparators')
    compare.update(invariants)
    for fixture, records in fixtures.values():
        tensors = {r['name'] for r in records if r['kind'] == 'tensor'}
        if not set(compare) <= tensors or not all(callable(fn) for fn in compare.values()):
            raise RuntimeError('invalid official validation outputs or comparators')
        for name in compare:
            if not (fixture / 'out' / (name + '.pt')).is_file():
                raise RuntimeError(f'missing trusted validation tensor: {name}')
    private = candidate.parent / 'isolated_worker'
    partition = arguments.get('_fixture_partition')
    if partition:
        private.mkdir(mode=0o700, exist_ok=True)
        private = private / partition
    private.mkdir(mode=0o700)
    runtime_tree = isolation.copy_runtime(private / 'runtime')
    context = private / 'context'
    context.mkdir()
    metadata = {'specs': {key: records for key, (_, records) in fixtures.items()}}
    if partition:
        metadata['fixture_partition'] = partition
    common.write_json(context / 'worker.json', metadata)
    common.write_json(context / 'case.json', case)
    for name in ('certification.json', 'ref_performance.json'):
        shutil.copyfile(sealed / name, context / name)
    source = case_dir / 'public'
    public = context / 'public'
    public.mkdir()
    # Public config/helper closure, excluding the original Candidate seed tree.
    for p in source.glob('*.py'):
        shutil.copyfile(p, public / p.name)
    shutil.copyfile(case_dir / case['comparator'], public / Path(case['comparator']).name)
    validation = Validation(private, fixtures, list(compare), compare)
    try:
        profile = common._environment_profile(case)
        devices = isolation.device_mapping(runtime)
        metadata['devices'] = devices
        sys.path.insert(0, str(common.BENCHMARK_ROOT))
        from modes._shared.profile_identity import collect as collect_device_identity
        device_identity_before = collect_device_identity(devices)
        metadata['device_identity'] = device_identity_before
        common.write_json(context / 'worker.json', metadata)
        view = isolation.View().system().toolchain(profile, runtime,
            devices=devices['physical'] if devices else None)
        if profile.get('platform') != 'cpu':
            # Resolve/verify the compiler's pin on the trusted side. The worker must
            # not write the installed resolver's shared lock/cache or see ISA docs.
            from pypto.runtime.pto_isa import ensure_pto_isa_root
            pinned = Path(ensure_pto_isa_root())
            from simpler_setup.pto_isa import ensure_pto_isa_root as ensure_runtime_isa
            if Path(ensure_runtime_isa()).resolve() != pinned.resolve():
                raise RuntimeError('PyPTO and runtime require different pinned ISA roots')
            metadata['pto_isa_root'] = str(pinned)
            revision = subprocess.run(['git', '-C', str(pinned), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True, timeout=5)
            status = subprocess.run(['git', '-C', str(pinned), 'status', '--porcelain=v1',
                                     '--untracked-files=normal'], capture_output=True, text=True, timeout=5)
            metadata['pto_isa_revision'] = (revision.stdout.strip()
                if revision.returncode == status.returncode == 0 and not status.stdout.strip() else None)
            common.write_json(context / 'worker.json', metadata)
            for headers in ('include', 'pkg_inc'):
                if (pinned / headers).is_dir():
                    view.add(pinned / headers)
        isolation.runtime_mounts(view, runtime_tree)
        view.add(context / 'case.json', case_dir / 'case.json')
        view.add(public, source)
        view.add(sealed / 'evaluate.py', sealed / 'evaluate.py')
        for name in ('certification.json', 'ref_performance.json'):
            view.add(context / name, sealed / name)
        for fixture_id, (fixture, _) in fixtures.items():
            view.add(fixture / 'in')
            view.add(fixture / 'manifest.json')
        view.add(common.BENCHMARK_ROOT / 'environments' / (case['environment'] + '.json'))
        view.add(context / 'worker.json', '/ptobench/worker.json')
        view.add(validation.socket_path, '/ptobench/validation.sock')
        view.add(validation.wire, '/ptobench/wire', write=True)
        view.add(candidate)  # frozen source is readonly, including tuning/contract.
        artifacts = candidate / '.benchmark_outputs'
        artifacts.mkdir(exist_ok=True)
        view.add(artifacts, write=True)
        result = private / 'result'
        result.mkdir()
        view.add(result, '/ptobench/result', write=True)
        env = isolation.clean_environment({**os.environ, **updates})
        # Canonical Case profile settings must apply before Candidate import, even
        # for keys outside the generic inherited-environment allowlist.
        env.update(updates)
        python_path = [str(common.BENCHMARK_ROOT), str(common.BENCHMARK_ROOT / 'evaluator')]
        if env.get('PYTHONPATH'):
            python_path.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = os.pathsep.join(python_path)
        sys.path.insert(0, str(common.BENCHMARK_ROOT))
        from modes._shared.profile_environment import isolated_environment
        env = isolated_environment(env, devices)
        if metadata.get('pto_isa_root'):
            from worker_startup import sitecustomize_source
            startup = private / 'python_startup'
            startup.mkdir()
            (startup / 'sitecustomize.py').write_text(sitecustomize_source())
            view.add(startup, '/ptobench/python_startup')
            env['PYTHONPATH'] = '/ptobench/python_startup:' + env['PYTHONPATH']
        env.update(PTOBENCH_WORKER='1', PTOBENCH_WORKER_METADATA='/ptobench/worker.json',
                   PTOBENCH_VALIDATION_SOCKET='/ptobench/validation.sock',
                   PTOBENCH_WORKER_WIRE='/ptobench/wire')
        # The environment was already bootstrapped by the trusted evaluate entry.
        mode = os.environ.get('PYPTO_BENCH_PROFILE', 'clean')
        argv = [sys.executable, str(common.BENCHMARK_ROOT / 'evaluator/evaluate.py'),
                '--case', case['case_id'], '--candidate', str(candidate), '--phase', arguments['phase'],
                '--profile-mode', mode, '--result', '/ptobench/result/summary.json']
        if arguments['platform'] is not None:
            argv.extend(['--platform', arguments['platform']])
        sys.path.insert(0, str(common.BENCHMARK_ROOT))
        if mode != 'clean' and os.environ.get('PTOBENCH_SKIP_CORRECTNESS') == '1':
            argv.append('--skip-correctness')
        common.write_json(private / 'filesystem_policy.json', view.manifest())
        diagnostics = _worker_diagnostics_directory(
            candidate, case['fixture_ids'][0], result)
        with _open_worker_log(diagnostics / 'worker_stdout.log') as stdout, \
                _open_worker_log(diagnostics / 'worker_stderr.log') as stderr:
            completed = subprocess.run(view.argv(argv, cwd=candidate, environment=env),
                                       stdout=stdout, stderr=stderr)
        device_identity_after = collect_device_identity(devices)
        worker_record, _ = _retain_worker_diagnostics(
            case['fixture_ids'][0], completed.returncode, result, diagnostics)
        if not (result / 'summary.json').is_file():
            raise RuntimeError(_worker_failure_message(worker_record, diagnostics))
        # Never trust a worker PASS without host-side output validation.
        summary = safe_json(result / 'summary.json')
        runs = summary['runs']
        if len(runs)!=1 or runs[0].get('fixture_id') not in fixtures:
            raise ValueError('worker fixture result identity is incomplete')
        requires_compare = mode == 'clean' or os.environ.get('PTOBENCH_SKIP_CORRECTNESS') != '1'
        if requires_compare and (validation.failed or validation.validated != set(case['fixture_ids'])):
            for run in runs:
                detail = validation.error or run.get('error') or 'was not performed'
                run.update(passed=False, error=f'trusted output validation: {detail}')
        manifest = summary.get('build_manifest', [])
        if (not isinstance(manifest, list) or any(not isinstance(item, dict) for item in manifest)
                or (len(runs) > 1 and (len(manifest) != len(runs)
                    or {item.get('fixture_id') for item in manifest} != set(case['fixture_ids'])
                    or any(type(item.get('invocation')) is not int or item['invocation'] != 0 for item in manifest)))):
            raise ValueError('worker fixture build manifest is incomplete')
        builds = {item['fixture_id']:item['build_dir'] for item in manifest}
        if len(runs) > 1 and len(set(builds.values())) != len(runs):
            raise ValueError('worker fixtures share a build directory')
        for run in runs:
            # The trusted parent binds persisted input bytes for single-input
            # Cases too; never accept an identity supplied by the worker.
            run['input_identity_sha256']=common.fixture_input_identity(sealed/'fixtures'/run['fixture_id'])
            from modes._shared.profile_identity import attach as attach_device_identity
            attach_device_identity(run, device_identity_before, device_identity_after)
            build = builds.get(run['fixture_id']) if manifest else summary.get('build_dir')
            if build:
                if not Path(build).resolve().is_relative_to(artifacts.resolve()):
                    raise ValueError('worker build directory is outside its artifacts')
                if len(runs) != 1 or run.get('fixture_id') not in fixtures:
                    raise ValueError('diagnostic worker stream fixture binding is ambiguous')
                run['_build_dir'] = build
        return runs
    finally:
        validation.close()


@contextlib.contextmanager
def bridge(fixture_id):
    """Worker-only hooks: keep original dispatch/readback/timing sequence."""
    import golden.runner as runner
    originals = {name: getattr(runner, name) for name in ('_require_files', '_compute_golden', '_validate')}
    def require_inputs(data_dir, required):
        return originals['_require_files'](data_dir, [(sub, name) for sub, name in required if sub != 'out'])
    def expected_marker(*args, **kwargs):
        # Non-None triggers the existing correctness dispatch and resident
        # readback; no expected values or private tensors enter this process.
        return {}
    def validate(tensor_specs, tensors, golden_outputs, rtol, atol, compare_fn, scalar_specs_eff=None):
        wire = Path(os.environ['PTOBENCH_WORKER_WIRE'])
        outputs = {}
        for i, spec in enumerate(s for s in tensor_specs if s.is_output):
            value = tensors[spec.name].detach().cpu().contiguous()
            leaf = f'{i}.bin'
            value.reshape(-1).view(torch.uint8).numpy().tofile(wire / leaf)
            outputs[spec.name] = {'file': leaf, 'shape': list(value.shape), 'dtype': str(value.dtype)}
        with socket.socket(socket.AF_UNIX) as connection:
            connection.connect(os.environ['PTOBENCH_VALIDATION_SOCKET'])
            connection.sendall((json.dumps({'fixture': fixture_id, 'outputs': outputs}) + '\n').encode())
            reply = json.loads(connection.makefile('rb').readline(1024 * 1024))
        if not reply['passed']:
            raise AssertionError(reply.get('error', 'trusted validation failed'))
    runner._require_files = require_inputs
    # Agent profiles retain the pre-existing no-validation/readback policy.
    if os.environ.get('PTOBENCH_SKIP_CORRECTNESS') != '1':
        runner._compute_golden = expected_marker
        runner._validate = validate
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(runner, name, value)
