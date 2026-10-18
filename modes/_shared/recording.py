"""Retain native agent events and bind native session files to one Bench Run."""
import hashlib
import json
import os
from pathlib import Path
import re

import common
import file_tree
from modes._shared.config import ConfigError


def read_bytes(path):
    with file_tree.open_file(path) as fd, os.fdopen(os.dup(fd), 'rb') as stream:
        return stream.read()


def option(argv, flag):
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(flag + '='):
            return arg[len(flag) + 1:]
    return None


def codex_override(argv, key):
    value = None
    for index, arg in enumerate(argv):
        assignment = (argv[index + 1] if arg in ('-c', '--config') and index + 1 < len(argv)
                      else arg.split('=', 1)[1] if arg.startswith('--config=') else '')
        if assignment.startswith(key + '='):
            value = assignment[len(key) + 1:].strip().strip('"').strip("'")
    return value


def config_directory(profile, environment=None):
    env = {**(os.environ if environment is None else environment), **profile.get('env', {})}
    if profile['backend'] == 'codex':
        return Path(env.get('CODEX_HOME', str(Path.home() / '.codex'))).expanduser()
    # The monorepo provider wrapper sets this path before exec'ing native Pi.
    if profile['argv'][0] == '{repo_root}/benchmarks/mig_workflow/agent_provider/launch.py':
        return Path.home() / '.config/pypto-agent-provider/pi'
    return Path(env.get('PI_CODING_AGENT_DIR', str(Path.home() / '.pi/agent'))).expanduser()


def host_settings(profile):
    path = config_directory(profile) / ('config.toml' if profile['backend'] == 'codex' else 'settings.json')
    try:
        raw = read_bytes(path.resolve())
    except FileNotFoundError:
        return {}
    if profile['backend'] == 'codex':
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        return tomllib.loads(raw.decode())
    return json.loads(raw)


def prepare_profile(profile):
    """Enable native recording for every Codex/Pi profile, including custom ones."""
    backend = profile['backend']
    if backend not in ('codex', 'pi'):
        return profile
    argv = list(profile['argv'])
    forbidden = ('--ephemeral', 'resume', 'fork') if backend == 'codex' else ('--no-session', '--continue', '--resume', '--session', '--session-id', '--fork', '-c', '-r')
    if any(arg.split('=')[0] in forbidden for arg in argv[1:]):
        raise ConfigError('experiments require a fresh persistent native agent session')
    additions = []
    settings = {} if profile.get('mode_provider') else host_settings(profile)
    if backend == 'codex':
        selected_profile = option(argv, '--profile') or option(argv, '-p')
        if selected_profile:
            settings = {**settings, **settings.get('profiles', {}).get(selected_profile, {})}
    if backend == 'codex':
        if '--json' not in argv:
            additions += ['--json']
        model_override = codex_override(argv, 'model')
        model = option(argv, '--model') or option(argv, '-m') or model_override or profile.get('model') or settings.get('model')
        if model and not (option(argv, '--model') or option(argv, '-m') or model_override):
            additions += ['--model', model]
        # Explicit CLI overrides take precedence; native turn_context records
        # the effective settings, including any named configuration profile.
        reasoning = profile.get('reasoning') or settings.get('model_reasoning_effort')
        if reasoning and codex_override(argv, 'model_reasoning_effort') is None:
            additions += ['-c', 'model_reasoning_effort=' + json.dumps(reasoning)]
    else:
        if option(argv, '--mode') not in (None, 'json'):
            raise ConfigError('Pi experiment output must use native JSON events')
        if option(argv, '--session-dir') not in (None, '{session_root}/agent/native'):
            raise ConfigError('Pi session storage must belong to this Run')
        if not option(argv, '--mode'):
            additions += ['--mode', 'json']
        if not option(argv, '--session-dir'):
            additions += ['--session-dir', '{session_root}/agent/native']
        configured_model = profile.get('model') or settings.get('defaultModel')
        if not option(argv, '--model') and configured_model:
            provider, model = (configured_model.split('/', 1) if '/' in configured_model
                               else (settings.get('defaultProvider'), configured_model))
            if provider and not option(argv, '--provider'):
                additions += ['--provider', provider]
            additions += ['--model', model]
        model = option(argv, '--model') or profile.get('model') or settings.get('defaultModel')
        model_thinking = model.rsplit(':', 1)[-1] if isinstance(model, str) and ':' in model else None
        if model_thinking not in ('off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
            model_thinking = None
        thinking = profile.get('reasoning') or model_thinking or settings.get('defaultThinkingLevel', 'off')
        if not option(argv, '--thinking'):
            additions += ['--thinking', thinking]
    index = argv.index('{prompt}')
    argv[index:index] = additions
    return {**profile, 'argv': argv, 'recording': {'native_events': True,
            'native_sessions': True, 'config_directory': str(config_directory(profile))}}


def _json_lines(path):
    with file_tree.open_file(path) as fd, os.fdopen(os.dup(fd), 'r', errors='replace') as stream:
        for line in stream:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
            except ValueError:
                # Keep all original bytes, including a truncated last event.
                continue


def collect(root, profile, workspace, *, started, close_reason):
    backend = profile['backend']
    result = {'schema_version': 1, 'run_id': root.name, 'backend': backend,
              'close_reason': close_reason, 'native_session_ids': [], 'files': []}
    if backend not in ('codex', 'pi'):
        return {**result, 'status': 'unsupported_backend'}
    if not started:
        return {**result, 'status': 'not_started'}
    native = root / 'agent/native'
    with file_tree.open_directory(native, create=True):
        pass
    events = root / 'stdout.log'
    if events.exists():
        file_tree.atomic_bytes(root / 'agent/events.jsonl', read_bytes(events), mode=0o444)
    ids = set()
    if events.exists():
        for event in _json_lines(events):
            candidate = event.get('thread_id') if event.get('type') == 'thread.started' else event.get('id') if event.get('type') == 'session' else None
            if isinstance(candidate, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', candidate):
                ids.add(candidate)
    if backend == 'codex':
        store = (config_directory(profile) / 'sessions').resolve()
        candidates = [path for identity in sorted(ids) for path in store.glob(f'**/*{identity}*.jsonl')]
    else:
        candidates = list(native.glob('*.jsonl'))
    for path in sorted(set(candidates)):
        header = next(_json_lines(path), {})
        meta = header.get('payload', {}) if backend == 'codex' and header.get('type') == 'session_meta' else header
        identity = meta.get('id')
        if not isinstance(identity, str) or (ids and identity not in ids):
            continue
        if meta.get('cwd') != str(workspace):
            continue
        data = read_bytes(path)
        destination = native / path.name
        if destination != path:
            file_tree.atomic_bytes(destination, data, mode=0o444)
        else:
            with file_tree.open_file(destination) as fd:
                os.fchmod(fd, 0o444)
        result['native_session_ids'].append(identity)
        result['files'].append({'path': str(destination.relative_to(root)), 'source': str(path),
                                'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    result['status'] = 'recorded' if result['files'] else 'missing'
    result['events_path'] = 'agent/events.jsonl' if events.exists() else None
    if not result['files']:
        result['message'] = 'No matching native session was persisted; retain event stream and terminal logs for diagnosis.'
    return result
