#!/usr/bin/env python3
"""Original CANNBot resources, native sessions and opaque participant records."""
import hashlib
import grp
import json
import os
from pathlib import Path
import sys
import shutil

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluator')]

from modes._shared.adapters import BackendUnavailable, get_adapter
from modes._shared.config import ConfigError, load_profile, render
from modes._shared import recording, provenance, provider
import isolation
from modes.cannbot import resources


def preflight(request):
    optimizer = request['experiment']['optimizer']
    if optimizer['backend'] not in ('pi', 'codex'):
        raise ConfigError('CANNBot supports Pi and Codex backends')
    profile = recording.prepare_profile(provider.configure(
        load_profile(optimizer['backend'], optimizer['profile']), Path(__file__).parent))
    info = get_adapter(optimizer['backend']).preflight(profile)
    # Only check resource availability; method API/profile usage is its own work.
    lock = resources.manifest()
    if any(not (resources.DIRECTORY / 'upstream' / name).is_file() for name in lock['files']):
        raise ConfigError('CANNBot upstream resources are unavailable')
    return {'schema_version': 1, 'profile': profile, 'backend': info.to_dict(),
            'optimizer_metadata': {'command': profile['argv'], 'tool_version': info.version}}


def prepare(request):
    context, ready = request['context'], request['preflight']
    root, candidate = Path(context['session_root']), Path(context['workspace'])
    prepared = ready['profile']
    gateway = (isolation.enabled() and prepared['backend'] == 'pi'
               and bool(prepared.get('mode_provider')))
    ready = {**ready, 'profile': provider.prepare(prepared, root, gateway=gateway)}
    extension = None
    if ready['profile']['backend'] == 'pi':
        share_gid = grp.getgrnam('ptobench-share').gr_gid
        extension_directory = root / 'agent/pi-tools'
        extension_directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        os.chown(extension_directory, -1, share_gid)
        extension_directory.chmod(0o750)
        extension = extension_directory / 'cannbot-extension.ts'
        shutil.copyfile(resources.DIRECTORY / 'bridge/pi_task.ts', extension)
        os.chown(extension, -1, share_gid)
        extension.chmod(0o440)
    upstream = root / 'method/resources/upstream'
    lock = resources.freeze(upstream)
    implementation = {}
    for relative in ('runner.py', 'runtime.py', 'tools.py', 'resources.py', 'entry.md', 'mode.yaml', 'bridge/pi_task.ts'):
        data = (resources.DIRECTORY / relative).read_bytes()
        target = root / 'method/resources/integration' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o444)
        implementation[relative] = hashlib.sha256(data).hexdigest()
    work = root / 'method/work'
    work.mkdir(parents=True, exist_ok=True)
    entry = Path(__file__).with_name('entry.md').read_text()
    prompt = f"""Optimize Case {context['case']['case_id']} within the experiment budget.
Minimize verified latency while preserving the public contract and correctness.
Candidate workspace: {candidate}
Edit Candidate source under `candidate/` and tuning options in `candidate/tuning.json`.
Entry: {context['case']['module']}::{context['case']['entry']}
Read .benchmark_context/README.md for the Case contract, Bench CLI and result locations.
Read .benchmark_context/tuning_capabilities.json for this Case's editable compiler and runtime scheduling options, allowed values and restrictions.
The compiler defaults remain PYPTO + false, and the complete defaults vary by Case.
After changing source or settings, use ptobench evaluate . to check correctness and clean latency.
PATH and the Bench Session environment are already set.
Hardware budget: {context['budget']['max_hardware_actions']} accepted exploration actions,
shared by all workspaces and Agent conversations in this Bench Session.
Frozen upstream root: {upstream}
Original main role: {upstream / resources.ROLES['pypto-op-orchestrator']}
Method work directory: {work}
    {entry}"""
    if request['experiment']['optimizer'].get('id', '').startswith('smoke_'):
        prompt += """
Smoke-test boundary: this is a flow check only. Run exactly one clean `ptobench evaluate .`,
confirm correctness and the effective tuning defaults, then stop and report the result.
Do not dispatch profiling, fork, or performance-tuning stages in this smoke session.
"""
    config = {'schema_version': 1, 'run_root': str(root), 'workspace': str(candidate),
              'profile': ready['profile'], 'upstream': str(upstream), 'roles': resources.ROLES,
              'skills': resources.skills(upstream), 'python': sys.executable,
              'agent_extension': str(extension) if extension else None,
              'prompt_path': str(root / 'prompt.txt'), 'work_dir': str(work),
              'hardware_budget': context['budget']['max_hardware_actions']}
    _, additions = render(ready['profile'], candidate, root, prompt)
    configuration = provenance.freeze_agent(root, ready['profile'], candidate, {**os.environ, **additions})
    return {'schema_version': 1, 'records': {
        'prompt.txt': prompt, 'profile.json': ready['profile'],
        'mode.json': {'id': 'cannbot', 'entry': entry, 'entry_sha256': hashlib.sha256(entry.encode()).hexdigest(),
                      'upstream_commit': lock['commit'], 'integration_files': implementation},
        'backend.json': {'backend': ready['backend'], 'cwd': str(candidate),
                         'profile_sha256': ready['profile']['sha256'],
                         'argv_source': 'method/dispatches/*/dispatch.json'},
        'agent/configuration.json': configuration,
        'method/config.json': config, 'method/upstream.json': lock},
        'archives': {'cannbot_resources': 'method/resources', 'agent_configuration': 'agent/config'}}


def run(request):
    from modes.cannbot.runtime import run_dispatch, load_config
    config = load_config(Path(request['context']['session_root']) / 'method/config.json')
    os.environ.update(provider.credentials(config['profile']))
    result = run_dispatch(config, 'pypto-op-orchestrator', Path(config['prompt_path']).read_text(),
                          parent=None, tool_call_id=None, forward=True)
    return_code = result['exit_code']
    raise SystemExit(return_code if return_code and return_code >= 0 else 1 if result['is_error'] else 0)


def collect(request):
    from modes.cannbot.runtime import collect_dispatches
    root = Path(request['context']['session_root'])
    result = collect_dispatches(root, request['preflight']['profile'], request['started'], request['close_reason'])
    archives = {key: value for key, value in (
        ('cannbot_dispatches', 'method/dispatches'), ('cannbot_work', 'method/work'),
        ('cannbot_tools', 'method/tools')) if (root / value).exists()}
    import file_tree
    for name in archives.values():
        for relative, entry in file_tree.scan_tree(root / name, reject_special=True).items():
            if entry['kind'] == 'file':
                with file_tree.open_file(root / name / relative) as descriptor:
                    os.fchmod(descriptor, entry['mode'] & ~0o222)
    return {'schema_version': 1, 'records': {'agent/session.json': result}, 'archives': archives, 'recording': result}


def main():
    try:
        operation, filename = sys.argv[1:]
        request = json.loads(Path(filename).read_text())
        value = {'preflight': preflight, 'prepare': prepare, 'run': run, 'collect': collect}[operation](request)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        known = isinstance(error, (ConfigError, BackendUnavailable))
        print(json.dumps({'schema_version': 1, 'error_code': getattr(error, 'error_code', 'infrastructure_error'),
                          'message': str(error) if known else 'CANNBot participant operation failed'}))
        return 2 if getattr(error, 'error_code', None) == 'invalid_request' else 5


if __name__ == '__main__':
    raise SystemExit(main())
