#!/usr/bin/env python3
"""Basic single-Agent lifecycle with Wiki and read-only Analyzer tools."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import grp

sys.dont_write_bytecode = True
DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path[:0] = [str(DIRECTORY), str(ROOT), str(ROOT / 'evaluator')]

import isolation
from modes._shared.adapters import BackendUnavailable, get_adapter
from modes._shared.config import ConfigError, load_profile, render
from modes._shared import provenance, recording, provider

# importlib accepts the Mode directory's hyphen without an invalid import statement.
resources = importlib.import_module('modes.basic_wiki-tool.resources')
make_prompt = importlib.import_module('modes.basic_wiki-tool.prompt').make_prompt
tools = importlib.import_module('modes.basic_wiki-tool.tools')
from analyzer_adapter import AnalyzerAdapter


def preflight(request):
    optimizer = request['experiment']['optimizer']
    if optimizer['backend'] != 'pi':
        raise ConfigError('basic_wiki-tool supports only Pi')
    resources.inventory(ROOT / 'pto_wiki/opt_wiki')
    profile = recording.prepare_profile(provider.configure(
        load_profile(optimizer['backend'], optimizer['profile']), DIRECTORY))
    info = get_adapter(optimizer['backend']).preflight(profile)
    return {'schema_version': 1, 'profile': profile, 'backend': info.to_dict(),
            'optimizer_metadata': {'command': profile['argv'], 'tool_version': info.version}}


def prepare(request):
    context = request['context']
    run_root, workspace = Path(context['session_root']), Path(context['workspace'])
    ready = request['preflight']
    gateway = isolation.enabled() and bool(ready['profile'].get('mode_provider'))
    profile = provider.prepare(ready['profile'], run_root, gateway=gateway)
    wiki = resources.freeze_wiki(ROOT / 'pto_wiki/opt_wiki', run_root, workspace)
    instructions = (DIRECTORY / 'instructions.md').read_bytes()
    prompt = make_prompt(request['experiment'], context, context['case'], instructions.decode(),
                         Path(wiki['visible_path']) / 'INDEX.md')
    extension_root = run_root / 'agent/pi-tools'
    extension_root.mkdir(parents=True, exist_ok=True)
    gid = grp.getgrnam('ptobench-share').gr_gid
    os.chown(extension_root, -1, gid)
    extension_root.chmod(0o750)
    extension = extension_root / 'pi_analyzer.ts'
    shutil.copyfile(DIRECTORY / 'bridge/pi_analyzer.ts', extension)
    manifest = json.loads((DIRECTORY / 'analyzer_manifest.json').read_text())
    specs = extension_root / 'pto-analyzer-specs.json'
    specs.write_text(json.dumps(manifest['specifications'], indent=2) + '\n')
    for path in (extension, specs):
        os.chown(path, -1, gid)
        path.chmod(0o440)
    resources.readonly(run_root / 'method/resources')
    argv, additions = render(profile, workspace, run_root, prompt)
    configuration = provenance.freeze_agent(run_root, profile, workspace, {**os.environ, **additions})
    identities = {str(p.relative_to(DIRECTORY)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [DIRECTORY / n for n in ('runner.py', 'prompt.py', 'instructions.md', 'resources.py',
                            'tools.py', 'analyzer_adapter.py', 'analyzer_manifest.json', 'mode.yaml', 'bridge/pi_analyzer.ts')]}
    mode = {'id': 'basic_wiki-tool', 'instructions': instructions.decode(), 'files': identities,
            'analyzer': manifest['files']}
    return {'schema_version': 1, 'records': {
        'prompt.txt': prompt, 'profile.json': profile, 'mode.json': mode,
        'method/wiki.json': wiki, 'method/analyzer_identity.json': manifest,
        'backend.json': {'backend': ready['backend'], 'argv': argv, 'cwd': str(workspace),
                         'profile_sha256': profile['sha256']},
        'agent/configuration.json': configuration},
        'archives': {'agent_configuration': 'agent/config', 'wiki_resources': 'method/resources',
                     'analyzer_extension': 'agent/pi-tools/pi_analyzer.ts',
                     'analyzer_specifications': 'agent/pi-tools/pto-analyzer-specs.json'}}


def run(request):
    import agent_launch
    profile = request['preflight']['profile']
    isolation.require_agent_isolation(profile)
    context = request['context']
    run_root = Path(context['session_root'])
    if profile.get('mode_provider'):
        profile = json.loads((run_root / 'profile.json').read_text())
    prompt = (run_root / 'prompt.txt').read_text()
    argv, additions = render(profile, context['workspace'], run_root, prompt)
    environment = {**os.environ, **additions, **provider.credentials(profile)}
    server = None
    previous = {}
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, stop)
        server = tools.ToolServer(AnalyzerAdapter(run_root), run_root / 'agent/analyzer.sock',
                                  run_root / 'method/analyzer_calls')
        argv, environment = isolation.agent_command(
            argv, cwd=context['workspace'], environment=environment,
            workspace=context['workspace'], session_root=run_root, backend='pi',
            pi_tool_extension=run_root / 'agent/pi-tools/pi_analyzer.ts',
            pi_tool_socket=server.path)
        return agent_launch.run_process(argv, cwd=context['workspace'], environment=environment)
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if server is not None:
                server.close()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def collect(request):
    context = request['context']
    run_root = Path(context['session_root'])
    result = recording.collect(run_root, request['preflight']['profile'], Path(context['workspace']),
                               started=request['started'], close_reason=request['close_reason'])
    tools.interrupt_unfinished(run_root / 'method/analyzer_calls')
    (run_root / 'agent/analyzer.sock').unlink(missing_ok=True)
    wiki = json.loads((run_root / 'method/wiki.json').read_text())
    verification = resources.verify_wiki(wiki)
    if not verification['unchanged']:
        result = {**result, 'status': 'missing', 'reason': 'Wiki copy integrity failed'}
    archives = {name: relative for name, relative in (
        ('agent_native_archive', 'agent/native'), ('agent_events_archive', 'agent/events.jsonl'),
        ('analyzer_calls', 'method/analyzer_calls'), ('analyzer_registration', 'method/analyzer_registration.json'))
        if (run_root / relative).exists()}
    for relative in archives.values():
        path = run_root / relative
        if path.is_dir():
            resources.readonly(path)
        else:
            path.chmod(0o440)
    return {'schema_version': 1, 'records': {'agent/session.json': result,
             'method/wiki_verification.json': verification}, 'archives': archives, 'recording': result}


def main():
    try:
        operation, filename = sys.argv[1:]
        request = json.loads(Path(filename).read_text())
        result = {'preflight': preflight, 'prepare': prepare, 'run': run, 'collect': collect}[operation](request)
        if operation == 'run':
            return result if type(result) is int else 0
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        known = isinstance(error, (ConfigError, BackendUnavailable))
        print(json.dumps({'schema_version': 1, 'error_code': getattr(error, 'error_code', 'infrastructure_error'),
                          'message': str(error) if known else 'basic_wiki-tool participant operation failed'}))
        return 2 if getattr(error, 'error_code', None) == 'invalid_request' else 5


if __name__ == '__main__':
    raise SystemExit(main())
