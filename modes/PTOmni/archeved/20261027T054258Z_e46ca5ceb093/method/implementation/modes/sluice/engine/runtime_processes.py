"""Bind runtime host processes to phase-local SDK device startup identities.

The worker's ordered stderr identifies graph/timing lifetimes. SDK per-process
files remain the only timing source when present; startup sidecars add no spans
to cost analysis. PID reuse and incomplete phase/device coverage refuse mapping.
No cross-clock arithmetic, physical device identity or communication cause is
inferred.
"""
import hashlib
import json
import re

VERSION = 'runtime_host_processes_v1'
BASIS = 'sdk_ready_phase_scoped_pid_v1'
_READY = re.compile(r'\[chip_process pid=([1-9][0-9]*) dev=(0|[1-9][0-9]*)\] ready')
_PHASE = re.compile(r'\[pypto\] l3_swimlane_pass=(graph|timing) event=(begin|end)')
_DEVICE = 'chip.run.runner_run.device_wall'


def _require(condition, reason):
    if not condition:
        raise ValueError('runtime_process_mapping_'+reason)


def bind(streams, device_ids, spans):
    """Each startup must belong to one complete local device set per phase.

    Separate SDK files have no shared line order with stderr. Their device
    process PIDs are joined only within this fixture/build's trusted worker
    lifetime, requiring unique PIDs across phases. READY follows device init
    and precedes dispatch by the SDK contract; it supplies no timing anchor.
    """
    from profile_db._offline_tools.strace_timing import parse_spans
    _require(isinstance(device_ids,list) and device_ids
        and all(type(i) is int and i>=0 for i in device_ids)
        and len(set(device_ids))==len(device_ids), 'device_declaration_invalid')
    measured = {s.pid for s in spans if s.name==_DEVICE}
    candidates = [lines for lines in streams.values()
                  if any('[chip_process' in line or 'l3_swimlane_pass=' in line for line in lines)]
    _require(len(candidates)==1, 'startup_stream_missing_or_ambiguous')
    lines = candidates[0]
    phased = any('l3_swimlane_pass=' in line for line in lines)
    phase,events = (None if phased else 'single_pass'),[]
    ready,phase_devices = {},{}
    for line in lines:
        if 'l3_swimlane_pass=' in line:
            match = _PHASE.fullmatch(line.strip())
            _require(match is not None,'phase_record_invalid')
            name,event = match.groups()
            _require((event=='begin' and phase is None) or (event=='end' and phase==name),
                     'phase_lifetime_ambiguous')
            events.append((name,event))
            phase = name if event=='begin' else None
        if '[chip_process' in line:
            match = _READY.fullmatch(line.strip())
            _require(match is not None,'startup_record_invalid')
            pid,device = map(int,match.groups())
            _require(phase is not None,'startup_outside_phase')
            _require(pid not in ready,'pid_lifetime_ambiguous')
            _require(device in device_ids,'undeclared_device')
            ready[pid] = {'rank':device_ids.index(device),'device':device,'capture_phase':phase}
            used = phase_devices.setdefault(phase,set())
            _require(device not in used,'device_repeated_in_phase')
            used.add(device)
        # If chip.run appears in this same ordered stream it must follow its
        # startup. File-based SDK spans are independently joined below.
        for span in parse_spans([line]):
            if span.name==_DEVICE or (span.name=='chip.run' and span.depth==0 and span.pid in measured):
                _require(span.pid in ready and ready[span.pid]['capture_phase']==phase,
                         'span_outside_process_lifetime')
    if phased:
        _require(events==[('graph','begin'),('graph','end'),('timing','begin'),('timing','end')],
                 'phase_coverage_incomplete')
    expected_phases = {'graph','timing'} if phased else {'single_pass'}
    _require(set(phase_devices)==expected_phases
             and all(v==set(device_ids) for v in phase_devices.values()),'phase_device_coverage_incomplete')
    roots = {s.pid for s in spans if s.name=='chip.run' and s.depth==0}
    _require(measured==set(ready) and measured<=roots,'host_process_coverage_mismatch')
    return ready


def analyze(adapter,handle):
    from profile_db.runtime_adapter.storage import read_snapshot, resolve
    from profile_db._offline_tools.strace_timing import parse_spans
    from .profile_batches import validate_original
    validate_original(adapter.db,handle)
    conn,run = adapter.db.connection,handle.db_run_id
    present = {r[0] for r in conn.execute('SELECT kind FROM artifact WHERE run_id=?',[run]).fetchall()}
    _require({'runtime_host_manifest','runtime_worker_stream:stdout','runtime_worker_stream:stderr'}<=present,
             'worker_originals_unavailable')
    doc = json.loads(resolve(conn,run,'runtime_host_manifest').read_text())
    _require(doc.get('schema_version')==1 and doc.get('origin')=='isolated_fixture_worker_stdio'
             and doc.get('fixture_id')==handle.identity.fixture_id and doc.get('profile_mode')=='runtime'
             and doc.get('additional_dispatches')==0,'worker_binding_invalid')
    entries = doc.get('captures')
    _require(isinstance(entries,list) and len(entries)==2 and all(isinstance(e,dict) for e in entries)
             and {e.get('stream') for e in entries}=={'stdout','stderr'},'worker_binding_invalid')
    streams = {}
    for item in entries:
        data = resolve(conn,run,'runtime_worker_stream:'+item['stream']).read_bytes()
        _require(item.get('file')=='host.worker.'+item['stream']+'.log'
                 and type(item.get('bytes')) is int and len(data)==item['bytes']
                 and hashlib.sha256(data).hexdigest()==item.get('sha256'),'worker_binding_invalid')
        streams[item['stream']] = data.decode('utf-8',errors='replace').splitlines()
    host = read_snapshot(conn,run,'host')
    payload = host.get('payload',{})
    _require(host.get('status')=='ok' and payload.get('incomplete_records')==0
             and not any(d.get('total',0)>0 for d in payload.get('drops',{}).values()),'host_capture_incomplete')
    logs = [resolve(conn,run,kind).read_text(errors='replace').splitlines()
            for kind in sorted(present) if kind.startswith('host_log:')]
    spans = list(parse_spans([line for lines in logs for line in lines]))
    # Both the persisted projection and raw records must describe these PIDs.
    _require({s.pid for s in spans if s.name==_DEVICE}<=
             {r['pid'] for r in payload.get('rounds',[])},'host_projection_mismatch')
    ids = (handle.identity.device or {}).get('runtime',{}).get('device_ids')
    mapping = bind(streams,ids,spans)
    if handle.identity.rank_label!='single':
        rank = handle.identity.device.get('rank')
        _require(type(rank) is int and 0<=rank<len(ids)
                 and handle.identity.device.get('device')==ids[rank],'rank_declaration_mismatch')
    return {'mapping':mapping,'rank_mapping_verified':True,'rank_mapping_basis':BASIS,
            'rank_mapping_scope':'diagnostic_device_processes_local_runtime_device_order',
            'rank_mapping_reason':None,'parser_version':VERSION}
