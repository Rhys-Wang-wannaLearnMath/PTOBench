#!/usr/bin/env python3
"""Pi-backed PTOBench mode with frozen structured knowledge."""
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
MODE = Path(__file__).resolve().parent
ROOT = MODE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluator')]

from prompt import make_prompt
from modes._shared.adapters import BackendUnavailable, get_adapter
from modes._shared.config import ConfigError, load_profile, render
from modes._shared import provider, provenance, recording

KNOWLEDGE_SOURCE = MODE / 'resources/knowledge'
SKILLS_SOURCE = MODE / 'resources/skills'


def _resource_files(directory):
    if not directory.is_dir() or directory.is_symlink():
        raise ConfigError('SLUICE Lite resources are unavailable')
    files = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ConfigError('SLUICE Lite resources contain a link')
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix == '.pyc' or '__pycache__' in path.parts:
            continue
        files.append(path)
    return files


def _validate_sources():
    knowledge_files = _resource_files(KNOWLEDGE_SOURCE)
    expected = ('cards', 'reference', 'packs', 'index')
    if any(not (KNOWLEDGE_SOURCE / name).is_dir() for name in expected):
        raise ConfigError('SLUICE Lite structured knowledge is incomplete')
    for path in knowledge_files:
        if path.suffix == '.json':
            json.loads(path.read_text())
    skills = _resource_files(SKILLS_SOURCE)
    if not any(path.name == 'SKILL.md' for path in skills):
        raise ConfigError('SLUICE Lite adapted PyPTO skill is unavailable')


def preflight(request):
    _validate_sources()
    experiment = request['experiment']
    if experiment['optimizer']['backend'] != 'pi':
        raise ConfigError('SLUICE Lite requires the Pi backend')
    if experiment.get('method_options', {}):
        raise ConfigError('SLUICE Lite does not define method options')
    profile = recording.prepare_profile(provider.configure(
        load_profile('pi', experiment['optimizer']['profile'], root=MODE / 'profiles'),
        MODE, prefer_profile=True))
    info = get_adapter('pi').preflight(profile)
    return {'schema_version': 1, 'profile': profile, 'backend': info.to_dict(),
            'optimizer_metadata': {'command': profile['argv'], 'tool_version': info.version},
            'resources': {'knowledge_files': len(_resource_files(KNOWLEDGE_SOURCE))}}


def _freeze_tree(source, destination):
    _resource_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    destination.chmod(0o755)
    for path in sorted(destination.rglob('*'), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    destination.chmod(0o555)


def _resource_guide(session_root):
    resources = Path(session_root) / 'method/resources'
    return f'''# SLUICE Lite resources

The Session contains read-only resources:

- Structured mechanism knowledge: `{resources}/knowledge/`
- Adapted PyPTO optimization skill: `{resources}/skills/pypto-performance/SKILL.md`

The knowledge bundle contains `cards/`, `reference/`, `packs/`, and `index/`.
Search the compact index fields, then open the matching card and any relevant reference:

```bash
python - <<'PY'
import json
from pathlib import Path

needle = 'task granularity'
index = json.loads(Path('{resources}/knowledge/index/search.json').read_text())
for item in index['items']:
    fields = [item.get('id', ''), item.get('title', ''), item.get('summary', ''), *item.get('aliases', [])]
    if needle.lower() in ' '.join(fields).lower():
        print(item['id'], item.get('summary', ''))
PY
```

Cards describe possible mechanisms and their constraints; use the Case source and measured feedback to decide what applies. PTOBench profile artifacts are available under `.benchmark_outputs/<mode>/` and may be inspected directly when a diagnostic question needs them.
'''


def prepare(request):
    _validate_sources()
    context = request['context']
    run_root = Path(context['session_root'])
    ready = request['preflight']
    profile = provider.prepare(ready['profile'], run_root)
    instructions = (MODE / 'instructions.md').read_text()
    resources_root = run_root / 'method/resources'
    _freeze_tree(KNOWLEDGE_SOURCE, resources_root / 'knowledge')
    _freeze_tree(SKILLS_SOURCE, resources_root / 'skills')
    skill_text = '\n\n'.join(path.read_text() for path in _resource_files(SKILLS_SOURCE)
                              if path.name == 'SKILL.md')
    instructions = f'{instructions}\n\n# Always-on PyPTO skill\n\n{skill_text}'
    prompt = make_prompt(request['experiment'], context, instructions)
    argv, additions = render(profile, context['workspace'], run_root, prompt)
    configuration = provenance.freeze_agent(run_root, profile, Path(context['workspace']),
                                            {**os.environ, **additions})
    records = {
        'prompt.txt': prompt,
        'profile.json': profile,
        'mode.json': {'id': request['experiment']['mode'], 'instructions': instructions},
        'backend.json': {'backend': ready['backend'], 'argv': argv, 'cwd': context['workspace'],
                         'profile_sha256': profile['sha256']},
        'agent/configuration.json': configuration,
        'method/README.md': _resource_guide(run_root),
        'method/knowledge.json': {'enabled': True, 'kind': 'structured_mechanism_knowledge'},
        'method/skills.json': {'enabled': True, 'kind': 'adapted_pypto_optimization_skill'},
    }
    return {'schema_version': 1, 'records': records,
            'archives': {'agent_configuration': 'agent/config',
                         'structured_knowledge': 'method/resources/knowledge',
                         'adapted_skills': 'method/resources/skills'}}


def run(request):
    context = request['context']
    run_root = Path(context['session_root'])
    profile = request['preflight']['profile']
    if profile.get('mode_provider'):
        profile = json.loads((run_root / 'profile.json').read_text())
    prompt = (run_root / 'prompt.txt').read_text()
    argv, additions = render(profile, context['workspace'], run_root, prompt)
    os.execvpe(argv[0], argv, {**os.environ, **additions, **provider.credentials(profile)})


def collect(request):
    root = Path(request['context']['session_root'])
    result = recording.collect(root, request['preflight']['profile'], Path(request['context']['workspace']),
                               started=request['started'], close_reason=request['close_reason'])
    archives = {name: relative for name, relative in (
        ('agent_native_archive', 'agent/native'), ('agent_events_archive', 'agent/events.jsonl'))
        if (root / relative).exists()}
    return {'schema_version': 1, 'records': {'agent/session.json': result},
            'archives': archives, 'recording': result}


def main():
    try:
        operation, filename = sys.argv[1:]
        request = json.loads(Path(filename).read_text())
        result = {'preflight': preflight, 'prepare': prepare, 'run': run, 'collect': collect}[operation](request)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        known = isinstance(error, (ConfigError, BackendUnavailable))
        print(json.dumps({'schema_version': 1,
                          'error_code': getattr(error, 'error_code', 'infrastructure_error'),
                          'message': str(error) if known else 'SLUICE Lite participant operation failed'}))
        return 2 if getattr(error, 'error_code', None) == 'invalid_request' else 5


if __name__ == '__main__':
    raise SystemExit(main())
