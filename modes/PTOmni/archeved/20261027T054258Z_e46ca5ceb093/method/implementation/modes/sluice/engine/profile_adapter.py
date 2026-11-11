"""Bind public action diagnostics to internal, session-local ProfileDB handles.

Only public result/context JSON is read here. ProfileDB owns capture parsing.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .contracts import ProfileRef, digest
from .evidence_guard import opaque
from ..foundation.context import _has_symlink_component

MODES = {'runtime': 'runtime', 'pmu': 'hardware', 'compiler': 'compile'}
REGISTRATION_VERSION = 3


@dataclass(frozen=True)
class ProfileIdentity:
    case_id: str
    fixture_id: str | None
    program_ref: str
    source_digest: str
    device: object = None
    software: object = None
    shape: object = None
    dtype: object = None
    shape_dtype_contract_ref: str | None = None
    tuning: object = None
    capture_level: int | None = None
    profile_mode: str | None = None
    rank_label: str = 'single'
    invocation: int = 0
    build_ref: str | None = None


@dataclass(frozen=True)
class ProfileHandle:
    ref: str
    identity: ProfileIdentity
    action_id: str
    event_seq: int | None
    observation_ref: str
    db_run_id: int | None = None
    artifact_fingerprint: tuple = ()
    availability: str = 'unavailable'
    reason: str | None = None
    member_refs: tuple[str, ...] = ()
    registration_version: int = 0
    member_kind: str | None = None
    member_slots: tuple[str, ...] = ()

    def to_public_ref(self):
        return ProfileRef(self.ref, self.identity.case_id, self.identity.source_digest,
                          self.action_id, self.event_seq, self.identity.profile_mode,
                          self.observation_ref, self.availability, self.reason, self.member_refs,
                          self.member_kind, self.member_slots)


def read_json(path):
    if _has_symlink_component(path):
        raise ValueError('public_path_invalid')
    return json.loads(path.read_text())


def within(path, root):
    path = Path(path)
    if _has_symlink_component(path) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('public_path_invalid')
    if any(p in ('staging', 'sealed', 'sealed_assets', 'evaluator') for p in path.relative_to(root).parts):
        raise ValueError('public_path_invalid')
    return path


class PublicIdentityReader:
    def __init__(self, context):
        self.context = context

    def read(self, obs, result):
        root = self.context.public_context_root(Path(obs.workspace))
        environment = read_json(root / 'environment.json')
        mode = read_json(root / 'mode.json')
        case = read_json(root / 'case.json')
        if case.get('case_id') != self.context.case_id or not mode.get('id'):
            raise ValueError('identity_mismatch')
        runs = result.get('runs', [])
        fixtures = {r.get('fixture_id') for r in runs}
        if len(runs) != 1 or len(fixtures) != 1 or not next(iter(fixtures)):
            raise ValueError('identity_missing')
        # Byte normalization is deliberately limited to newlines. No fixture or
        # executable config is loaded; shape/dtype identify the public contract.
        contract = digest([p.read_text().replace('\r\n', '\n') for p in (
            within(root / 'case.json', root), within(root / 'config.py', root),
            within(root / 'comparator.json', root))])
        tuning = result.get('effective_tuning')
        diagnostics = runs[0].get('measurement_config', {}).get('tuning', {}).get('diagnostics', {})
        level = diagnostics.get('enable_chip_swimlane') if obs.profile_mode == 'runtime' else diagnostics.get('enable_pmu') if obs.profile_mode == 'pmu' else None
        if obs.profile_mode == 'compiler' and diagnostics.get('compile_profiling') is not True:
            raise ValueError('identity_missing')
        if obs.profile_mode == 'pmu' and (type(level) is not int or level <= 0):
            raise ValueError('identity_missing')
        tuning = {'effective': tuning, 'diagnostics': diagnostics} if tuning is not None else None
        device = None
        if environment.get('platform') is not None and environment.get('device') is not None:
            device = {k: environment.get(k) for k in ('platform', 'device', 'runtime')}
            device['runtime'] = {**(environment.get('runtime') or {}), **(case.get('runtime') or {})}
            if environment.get('platform') in ('a2a3', 'a5'):
                from ..profiling.device_identity import comparison_identity
                device['physical_identity'] = comparison_identity(
                    runs[0].get('device_identity'), obs.method_action_id)
                physical = device['physical_identity']
                if physical['status'] == 'verified':
                    expected = device['runtime'].get('device_ids', [environment['device']])
                    if [d['configured_device'] for d in physical['devices']] != expected:
                        raise ValueError('identity_mismatch')
        shape = {'contract':contract}
        catalog = root/'fixture_catalog.json'
        if catalog.is_file():
            doc = read_json(catalog)
            entries = [f for f in doc.get('fixtures', []) if f.get('fixture_id') in fixtures]
            input_id=entries[0].get('input_identity_sha256') if len(entries)==1 else None
            if (doc.get('schema_version') != 1 or len(entries) != 1
                    or not isinstance(input_id,str) or len(input_id)!=64
                    or any(c not in '0123456789abcdef' for c in input_id)
                    or input_id != runs[0].get('input_identity_sha256')):
                raise ValueError('identity_mismatch')
            shape['input_artifacts'] = input_id
        return ProfileIdentity(self.context.case_id, next(iter(fixtures)),
                               opaque('program', obs.source_before), obs.source_before,
                               device, result.get('software_identity'), shape,
                               {'contract': contract}, contract, tuning,
                               level if type(level) is int else None, obs.profile_mode)


class ProfileAdapter:
    def __init__(self, context, db=None):
        from .profile_core import profile_db_class
        ProfileDB = profile_db_class()
        self.context = context
        self.db = db if db is not None else ProfileDB(context.work_root / 'profile_db/profile.duckdb')
        if db is None:
            # Session databases belong to this adapter. Repair legacy physical
            # rows before restoring handles; failed runs remain intact and
            # cannot pass the independent physical-parser compatibility gate.
            from .controller import save
            save(context.work_root / 'profile_db/repair_status.json', self.db.repair_all())
        self.reader = PublicIdentityReader(context)
        self.root = context.work_root / 'profile_handles'
        self.handles = {}
        if self.root.exists():
            for path in sorted(self.root.glob('*.json')):
                value = read_json(path)
                value['identity'] = ProfileIdentity(**value['identity'])
                value['artifact_fingerprint'] = tuple(value['artifact_fingerprint'])
                value['member_refs'] = tuple(value.get('member_refs',()))
                value['member_slots'] = tuple(value.get('member_slots',()))
                handle = ProfileHandle(**value)
                self.handles[handle.ref] = handle

    def _save(self, handle):
        from .controller import save
        handle = replace(handle, registration_version=REGISTRATION_VERSION)
        save(self.root / (handle.ref.split(':')[1] + '.json'), asdict(handle))
        self.handles[handle.ref] = handle
        return handle

    def _cached_current(self, ref):
        handle = self.handles.get(ref)
        if (handle is None or handle.availability != 'available'
                or handle.registration_version != REGISTRATION_VERSION):
            return None
        def leaves(h):
            return [leaf for ref in h.member_refs for leaf in leaves(self.handles[ref])] if h.member_refs else [h]
        members = leaves(handle)
        if all(self.db.compatibility(h.db_run_id, h.db_run_id)['compatible'] for h in members):
            return handle
        return None

    def _attach_action_host(self, obs, run_id):
        """Single-process runtimes emit host records on the public CLI stream.

        Distributed runtimes already write host.*.log in the build. Use one
        source only; logs describe the action and must not be counted per rank.
        """
        if self.db.connection.execute("SELECT 1 FROM artifact WHERE run_id=? AND kind LIKE 'host_log:%' LIMIT 1", [run_id]).fetchone():
            return
        lines = []
        for name in ('stderr_path', 'stdout_path'):
            relative = getattr(obs, name)
            stream = self.context.session_root / relative
            if not stream.is_file():
                continue
            action = within(self.context.session_root / obs.result_path, self.context.work_root).parent
            stream = within(stream, action)
            if stream.name not in ('stderr.log', 'stdout.log'):
                raise ValueError('action_result_mismatch')
            lines.extend(line for line in stream.read_text(errors='replace').splitlines()
                         if any(marker in line for marker in ('[STRACE]', '[HOSTLOG_DROPS]', '[CLOCK_ANCHOR]')))
        if not any('[STRACE]' in line for line in lines):
            return
        directory = self.context.work_root/'profile_host'/opaque('evidence', obs.method_action_id).split(':')[1]
        target = within(directory/'host.action.log', self.context.work_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = '\n'.join(lines)+'\n'
        if target.exists() and target.read_text()!=text:
            raise ValueError('action_result_mismatch')
        target.write_text(text)
        self.db.attach(target, run_id=run_id, copy=self.db.path is not None)

    def _attach_device_identity(self, raw, root, run_id):
        if not any(r.get('device_identity') is not None for r in raw.get('runs', [])):
            return
        from profile_db.ingest.analysis import register_paths
        copied = []
        try:
            with self.db._writing() as conn:
                register_paths(conn, run_id,
                               {'device_identity_result': within(root / 'latest_result.json', root)},
                               copy=self.db.path is not None, copied=copied)
        except Exception:
            for path in copied:
                path.unlink(missing_ok=True)
            raise


    def _attach_worker_host(self, build, root, identity, run_id):
        """Use only both exact streams bound to this fixture and diagnostic build."""
        import hashlib
        from profile_db.ingest.analysis import register_paths
        manifest = within(build/'runtime_host_capture.json',root)
        doc = read_json(manifest)
        entries = doc.get('captures')
        if (doc.get('schema_version')!=1 or doc.get('origin')!='isolated_fixture_worker_stdio'
                or doc.get('fixture_id')!=identity.fixture_id or doc.get('profile_mode')!=identity.profile_mode
                or doc.get('build_path')!=build.relative_to(root).as_posix() or doc.get('additional_dispatches')!=0
                or not isinstance(entries,list) or len(entries)!=2
                or any(not isinstance(e,dict) for e in entries)
                or {e.get('stream') for e in entries}!={'stdout','stderr'}):
            raise ValueError('runtime_host_manifest_invalid')
        paths = {'runtime_host_manifest':manifest}
        existing_host = self.db.connection.execute(
            "SELECT 1 FROM artifact WHERE run_id=? AND kind LIKE 'host_log:%' LIMIT 1", [run_id]).fetchone()
        for entry in entries:
            name = 'host.worker.'+entry['stream']+'.log'
            if entry.get('file')!=name:
                raise ValueError('runtime_host_manifest_invalid')
            path = within(build/name,root)
            data = path.read_bytes()
            if type(entry.get('bytes')) is not int or len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry.get('sha256'):
                raise ValueError('runtime_host_manifest_invalid')
            # Keep startup/phase records even when the SDK supplies its own
            # per-process timing files. These sidecars never enter host cost
            # analysis, which must consume exactly one source of spans.
            paths['runtime_worker_stream:'+entry['stream']] = path
            if not existing_host:
                paths['host_log:'+name] = path
        copied = []
        try:
            with self.db._writing() as conn:
                register_paths(conn,run_id,paths,copy=self.db.path is not None,copied=copied)
        except Exception:
            for path in copied:
                path.unlink(missing_ok=True)
            raise
        self.db.refresh(run_id=run_id)

    def _validate_fixture_coverage(self, obs, raw):
        public = self.context.public_context_root(Path(obs.workspace))
        declared = read_json(public/'case.json').get('fixture_ids')
        if declared is None:  # Historical/synthetic public contexts.
            return
        runs = raw.get('runs', [])
        if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared)
                or len(runs) != len(declared) or {r.get('fixture_id') for r in runs} != set(declared)):
            raise ValueError('identity_mismatch')
        if len(declared) > 1:
            catalog = read_json(public/'fixture_catalog.json')
            fixtures = catalog.get('fixtures', [])
            if (catalog.get('schema_version') != 1 or len(fixtures) != len(declared)
                    or {f.get('fixture_id') for f in fixtures} != set(declared)):
                raise ValueError('identity_mismatch')
            identities = {f['fixture_id']:f.get('input_identity_sha256') for f in fixtures}
            if any(not isinstance(identities[r['fixture_id']], str)
                   or len(identities[r['fixture_id']]) != 64
                   or r.get('input_identity_sha256') != identities[r['fixture_id']] for r in runs):
                raise ValueError('identity_mismatch')

    def _attach_benchmark_host(self, raw, root, run_id, workspace):
        """Validate the owned retention manifest, then archive the exact log.

        Optional host evidence must not invalidate an otherwise valid clean
        measurement. A missing or invalid attachment stays unavailable.
        """
        import hashlib
        if raw.get('build_dir') is None:
            return
        build = within(Path(raw['build_dir']), root)
        manifest = build / 'benchmark_host_capture.json'
        if not manifest.is_file():
            return
        doc = read_json(within(manifest, root))
        captures = doc.get('captures')
        config = raw['runs'][0]['measurement_config']
        if (doc.get('schema_version') != 1 or doc.get('origin') != 'existing_benchmark_parser_input'
                or doc.get('additional_dispatches') != 0 or doc.get('includes_warmup') is not True
                or doc.get('cold_compile_certified') is not False
                or not isinstance(captures, list) or len(captures) != 1):
            raise ValueError('benchmark_host_manifest_invalid')
        item = captures[0]
        if (item.get('file') != 'host.benchmark_000.log'
                or item.get('rounds') != config.get('actual_rounds')
                or item.get('warmup') != config.get('actual_warmup')
                or type(item.get('distributed')) is not bool):
            raise ValueError('benchmark_host_manifest_invalid')
        path = within(build / item['file'], root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get('sha256'):
            raise ValueError('benchmark_host_manifest_invalid')
        self.db.attach(path, run_id=run_id, copy=self.db.path is not None)
        # Preserve parser arguments and the hashed public contract, allowing
        # clean process reconstruction after the original workspace is gone.
        from profile_db.ingest.analysis import register_paths
        public = self.context.public_context_root(workspace)
        paths = {'clean_host_manifest': manifest, **{
            'clean_public_' + kind: within(public / name, public)
            for kind, name in (('case', 'case.json'), ('config', 'config.py'),
                               ('comparator', 'comparator.json'))}}
        copied = []
        try:
            with self.db._writing() as conn:
                register_paths(conn, run_id, paths, copy=self.db.path is not None, copied=copied)
        except Exception:
            for target in copied:
                target.unlink(missing_ok=True)
            raise

    def register_evaluate_action(self, obs):
        ref = opaque('profile', self.context.session_id + ':clean:' + obs.method_action_id)
        cached = self._cached_current(ref)
        if cached is not None:
            return cached
        identity = ProfileIdentity(self.context.case_id, None, opaque('program', obs.source_before),
            obs.source_before, profile_mode='clean')
        handle = ProfileHandle(ref, identity, obs.method_action_id, obs.bench_event_seq, obs.result_path)
        try:
            result = obs.cli_result or {}
            if (obs.operation != 'evaluate' or obs.profile_mode is not None or obs.exit_code != 0
                    or obs.parse_status != 'valid' or result.get('status') != 'PASS'
                    or result.get('correctness') != 'PASS' or result.get('integrity_passed') is not True
                    or obs.source_before != obs.source_after or type(obs.bench_event_seq) is not int):
                return self._save(replace(handle, reason='evaluate_action_failed'))
            from ..foundation.profile_publication import public_result
            selected = self.context.validate_workspace(Path(obs.workspace))
            root = within(selected / '.benchmark_outputs/clean', selected)
            raw = public_result(self.context, obs)
            self._validate_fixture_coverage(obs, raw)
            if len(raw.get('runs', [])) > 1:
                from .profile_batches import register_batch
                return register_batch(self, obs, raw, root, handle, selected)
            return self._register_clean_payload(obs, raw, root, handle, root/'latest_result.json', selected)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            code = str(exc)
            reason = code if code in {'public_path_invalid', 'identity_missing', 'identity_mismatch',
                'action_result_mismatch'} else 'measurement_invalid'
        except Exception:
            reason = 'measurement_invalid'
        return self._save(replace(handle, reason=reason))


    def _register_clean_payload(self, obs, raw, root, handle, source, selected):
        identity = self.reader.read(replace(obs, profile_mode='clean'), raw)
        handle = replace(handle, identity=identity)
        required = asdict(identity)
        if any(v is None for k, v in required.items() if k not in ('capture_level','build_ref')):
            raise ValueError('identity_missing')
        report = self.db.ingest_benchmark(source, program=self.context.case_id,
            identity={**required, 'action_id': obs.method_action_id, 'event_seq': obs.bench_event_seq},
            receipt=obs.cli_result, allowed_root=source.parent, copy=self.db.path is not None)
        try:
            self._attach_benchmark_host(raw, root, report['run_id'], selected)
        except Exception:
            # Clean sample validity and optional host coverage are separate.
            # No attachment means no matching host facts in the public tool.
            from .controller import save
            save(self.context.work_root / 'profile_host' / (handle.ref.split(':')[1] + '.json'),
                 {'status': 'unavailable', 'reason': 'benchmark_host_attachment_invalid'})
        internal = self.db.get_profile_handle(report['run_id'])
        return self._save(replace(handle, db_run_id=report['run_id'],
            artifact_fingerprint=tuple(internal['artifact_fingerprint']), availability='available'))

    def register_profile_action(self, obs):
        ref = opaque('profile', self.context.session_id + ':' + obs.method_action_id)
        cached = self._cached_current(ref)
        if cached is not None:
            return cached
        identity = ProfileIdentity(self.context.case_id, None, opaque('program', obs.source_before),
                                   obs.source_before, profile_mode=obs.profile_mode)
        handle = ProfileHandle(ref, identity, obs.method_action_id, obs.bench_event_seq, obs.result_path)
        reason = 'profile_action_failed'
        try:
            result = obs.cli_result or {}
            if (obs.operation != 'profile' or obs.exit_code != 0 or obs.parse_status != 'valid'
                    or result.get('integrity_passed') is not True or obs.source_before != obs.source_after
                    or result.get('status') != 'PASS' or type(obs.bench_event_seq) is not int):
                return self._save(replace(handle, reason=reason))
            if obs.profile_mode not in MODES:
                return self._save(replace(handle, reason='mode_unsupported'))
            from ..foundation.profile_publication import public_result
            selected = self.context.validate_workspace(Path(obs.workspace))
            root = within(selected / '.benchmark_outputs' / MODES[obs.profile_mode], selected)
            public = read_json(root / 'latest_session_result.json')
            if any(public.get(k) != result.get(k) for k in ('event_seq', 'integrity_passed', 'status')):
                raise ValueError('action_result_mismatch')
            raw = public_result(self.context, obs)
            self._validate_fixture_coverage(obs, raw)
            if (raw.get('case_id') != self.context.case_id or raw.get('profiling_mode') != MODES[obs.profile_mode]
                    or raw.get('diagnostic_only') is not True):
                raise ValueError('identity_mismatch')
            if len(raw.get('runs', [])) > 1:
                from .profile_batches import register_batch
                return register_batch(self, obs, raw, root, handle, selected)
            return self._register_profile_payload(obs, raw, root, handle)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            code = str(exc)
            reason = code if code in {'public_path_invalid', 'capture_missing', 'capture_ambiguous', 'capture_partial',
                'identity_missing', 'identity_mismatch', 'action_result_mismatch'} else 'capture_missing'
        except Exception as exc:
            from profile_db.errors import IngestError
            reason = 'parse_failed' if isinstance(exc, IngestError) else 'infrastructure_error'
        return self._save(replace(handle, reason=reason))

    def _register_profile_payload(self, obs, raw, root, handle, build=None):
        if build is None:
            for key in ('artifact_dir', 'build_dir'):
                if raw.get(key) is None:
                    raise ValueError('capture_missing')
                within(Path(raw[key]), root)
            candidates = sorted(root.glob('build_output/*/dfx_outputs' if obs.profile_mode == 'runtime' else 'build_output/*'))
            if len(candidates) != 1:
                raise ValueError('capture_missing' if not candidates else 'capture_ambiguous')
            capture = within(candidates[0], root)
            build = capture.parent if obs.profile_mode == 'runtime' else capture
            if Path(raw['build_dir']) != build:
                raise ValueError('identity_mismatch')
        else:
            build = within(build, root)
            capture = within(build/'dfx_outputs' if obs.profile_mode == 'runtime' else build, root)
        # Reject links before any core parser can discover optional artifacts.
        for path in (capture.parent if obs.profile_mode == 'runtime' else capture).rglob('*'):
            within(path, root)
        build = capture.parent if obs.profile_mode == 'runtime' else capture
        if obs.profile_mode in ('runtime','pmu') and (build/'distributed_meta.json').is_file():
            return self._register_distributed(obs,raw,root,build,handle)
        identity = self.reader.read(obs, raw)
        identity = replace(identity,build_ref=opaque('program',(obs.method_action_id,str(build.relative_to(root)))))
        handle = replace(handle, identity=identity)
        required = asdict(identity)
        if any(v is None for k, v in required.items() if not (k == 'capture_level' and obs.profile_mode == 'compiler')):
            raise ValueError('identity_missing')
        meta = dict(program=self.context.case_id, identity={**required,
            'action_id': obs.method_action_id, 'event_seq': obs.bench_event_seq}, allowed_root=root,
            copy=self.db.path is not None)
        report = (self.db.ingest(capture, prune_after=False, **meta) if obs.profile_mode == 'runtime'
                  else self.db.ingest_diagnostic(capture, mode=obs.profile_mode, **meta))
        self._attach_device_identity(raw, root, report['run_id'])
        if obs.profile_mode == 'runtime' and any(doc['status']=='stale' for doc in
                self.db.get_profile_handle(report['run_id'])['analyses'].values()):
            self.db.refresh(run_id=report['run_id'])
        if obs.profile_mode == 'runtime':
            if (build/'runtime_host_capture.json').is_file():
                try:
                    self._attach_worker_host(build,root,identity,report['run_id'])
                except (OSError,ValueError,KeyError,TypeError):
                    pass  # Optional damaged host evidence cannot invalidate runtime work evidence.
            elif not raw.get('batch_projection'):
                self._attach_action_host(obs, report['run_id'])
        internal = self.db.get_profile_handle(report['run_id'])
        if obs.profile_mode == 'runtime' and internal['capture_level'] != identity.capture_level:
            raise ValueError('identity_mismatch')
        return self._save(replace(handle, db_run_id=report['run_id'],
            artifact_fingerprint=tuple(internal['artifact_fingerprint']), availability='available'))

    def _register_distributed(self, obs, raw, root, build, parent):
        import re
        metadata = read_json(build/'distributed_meta.json')
        devices = metadata.get('distributed_config',{}).get('device_ids')
        if not isinstance(devices,list) or not devices or len(set(devices)) != len(devices):
            raise ValueError('identity_missing')
        pattern = 'chip_swimlane_records.json' if obs.profile_mode == 'runtime' else 'pmu.csv'
        captures = {}
        dispatches = set()
        for file in sorted((build/'dfx_outputs').rglob(pattern)):
            relative = file.parent.relative_to(build/'dfx_outputs')
            if len(relative.parts)!=2 or not re.fullmatch(r'rank\d+',relative.parts[0]) or not re.fullmatch(r'd\d+',relative.parts[1]):
                raise ValueError('capture_ambiguous')
            rank,dispatch = int(relative.parts[0][4:]),int(relative.parts[1][1:])
            captures[(rank,dispatch)] = within(file.parent,root)
            dispatches.add(dispatch)
        expected = {(rank,dispatch) for rank in range(len(devices)) for dispatch in dispatches}
        if not expected or captures.keys()!=expected:
            raise ValueError('capture_partial')
        base = self.reader.read(obs,raw)
        build_ref = opaque('program',(obs.method_action_id,str(build.relative_to(root))))
        children = []
        for rank,dispatch in sorted(expected):
            label = f'rank{rank}_d{dispatch}'
            ref = opaque('profile',(parent.ref,label))
            identity = replace(base,device={**base.device,'device':devices[rank],'rank':rank},
                               rank_label=label,invocation=dispatch,build_ref=build_ref)
            meta = dict(program=self.context.case_id,identity={**asdict(identity),
                'action_id':obs.method_action_id,'event_seq':obs.bench_event_seq},
                allowed_root=root,copy=self.db.path is not None)
            report = (self.db.ingest(captures[(rank,dispatch)],rank_label=label,prune_after=False,**meta)
                      if obs.profile_mode=='runtime' else self.db.ingest_diagnostic(captures[(rank,dispatch)],mode='pmu',**meta))
            self._attach_device_identity(raw, root, report['run_id'])
            if obs.profile_mode=='runtime' and any(doc['status']=='stale' for doc in
                    self.db.get_profile_handle(report['run_id'])['analyses'].values()):
                self.db.refresh(run_id=report['run_id'])
            if obs.profile_mode=='runtime' and not children:
                for host in sorted((build/'dfx_outputs').glob('host.*.log')):
                    self.db.attach(within(host,root),run_id=report['run_id'],copy=self.db.path is not None)
                if (build/'runtime_host_capture.json').is_file():
                    try:
                        self._attach_worker_host(build,root,identity,report['run_id'])
                    except (OSError,ValueError,KeyError,TypeError):
                        pass  # Optional worker evidence must not discard valid rank captures.
                elif not raw.get('batch_projection'):
                    self._attach_action_host(obs, report['run_id'])
            internal = self.db.get_profile_handle(report['run_id'])
            if obs.profile_mode=='runtime' and internal['capture_level']!=identity.capture_level:
                raise ValueError('identity_mismatch')
            child = ProfileHandle(ref,identity,obs.method_action_id,obs.bench_event_seq,obs.result_path,
                report['run_id'],tuple(internal['artifact_fingerprint']),'available')
            children.append(self._save(child))
        return self._save(replace(parent,identity=replace(base,build_ref=build_ref),
            db_run_id=children[0].db_run_id,artifact_fingerprint=tuple(c.artifact_fingerprint for c in children),
            availability='available',member_refs=tuple(c.ref for c in children),member_kind='rank_dispatch',
            member_slots=tuple(c.identity.rank_label for c in children)))
