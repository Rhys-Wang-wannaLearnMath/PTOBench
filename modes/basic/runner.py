#!/usr/bin/env python3
"""Independent Basic participant: prepare, exec one native agent, retain history."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'evaluator'))

from modes.basic.prompt import make_prompt
from modes._shared.adapters import BackendUnavailable, get_adapter
from modes._shared.config import ConfigError, load_profile, render
from modes._shared import provenance, recording, knowledge, provider


def preflight(request):
    knowledge.validate(Path(__file__).parent)
    optimizer = request['experiment']['optimizer']
    profile = recording.prepare_profile(provider.configure(
        load_profile(optimizer['backend'], optimizer['profile']), Path(__file__).parent))
    info = get_adapter(optimizer['backend']).preflight(profile)
    return {'schema_version': 1, 'profile': profile, 'backend': info.to_dict(),
            'optimizer_metadata': {'command': profile['argv'], 'tool_version': info.version}}


def prepare(request):
    context = request['context']
    run_root = Path(context['session_root'])
    workspace = Path(context['workspace'])
    ready = request['preflight']
    profile = provider.prepare(ready['profile'], run_root)
    references = knowledge.freeze(Path(__file__).parent, run_root)
    instructions_path = Path(__file__).with_name('instructions.md')
    instructions = instructions_path.read_bytes()
    prompt = make_prompt(request['experiment'], context, context['case'], instructions.decode())
    argv, additions = render(profile, workspace, run_root, prompt)
    configuration = provenance.freeze_agent(run_root, profile, workspace, {**os.environ, **additions})
    mode = {'id': request['experiment']['mode'], 'instructions': instructions.decode(),
            'files': {'instructions.md': hashlib.sha256(instructions).hexdigest()}}
    return {'schema_version': 1, 'records': {
        'prompt.txt': prompt, 'profile.json': profile, 'mode.json': mode,
        'backend.json': {'backend': ready['backend'], 'argv': argv, 'cwd': str(workspace),
                         'profile_sha256': profile['sha256']},
        'agent/configuration.json': configuration,
        'method/README.md': knowledge.guide(run_root), 'method/knowledge.json': references},
        'archives': {'agent_configuration': 'agent/config', 'reference_knowledge': knowledge.SNAPSHOT}}


def run(request):
    context = request['context']
    run_root = Path(context['session_root'])
    profile = request['preflight']['profile']
    if profile.get('mode_provider'):
        profile = json.loads((run_root / 'profile.json').read_text())
    # Use the exact frozen prompt; exec preserves the owned PID/group and exit code.
    prompt = (run_root / 'prompt.txt').read_text()
    argv, additions = render(profile, context['workspace'], run_root, prompt)
    os.execvpe(argv[0], argv, {**os.environ, **additions, **provider.credentials(profile)})


def collect(request):
    context = request['context']
    run_root = Path(context['session_root'])
    result = recording.collect(run_root, request['preflight']['profile'], Path(context['workspace']),
                               started=request['started'], close_reason=request['close_reason'])
    archives = {name: relative for name, relative in (
        ('agent_native_archive', 'agent/native'), ('agent_events_archive', 'agent/events.jsonl'))
        if (run_root / relative).exists()}
    return {'schema_version': 1, 'records': {'agent/session.json': result}, 'archives': archives,
            'recording': result}


def main():
    try:
        operation, filename = sys.argv[1:]
        request = json.loads(Path(filename).read_text())
        result = {'preflight': preflight, 'prepare': prepare, 'run': run, 'collect': collect}[operation](request)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        known = isinstance(error, (ConfigError, BackendUnavailable))
        print(json.dumps({'schema_version': 1, 'error_code': getattr(error, 'error_code', 'infrastructure_error'),
                          'message': str(error) if known else 'Basic participant operation failed'}))
        return 2 if getattr(error, 'error_code', None) == 'invalid_request' else 5


if __name__ == '__main__':
    raise SystemExit(main())
