"""Freeze non-secret agent configuration and instruction sources before launch."""
import hashlib
import json
import os
from pathlib import Path
import re

import common
import file_tree
from modes._shared import recording

SECRET = re.compile(r'(?:^|_)(?:api_?key|auth_?token|access_?token|refresh_?token|password|secret|credential|authorization)(?:$|_)', re.I)


def redact(value):
    if isinstance(value, dict):
        return {key: '[REDACTED]' if SECRET.search(key) or key.lower() in {'apikey', 'token', 'auth', 'headers'} else redact(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r'\bBearer\s+[^\s"\']+|\bsk-[A-Za-z0-9_-]{10,}', '[REDACTED]', value)
    return value


def freeze_agent(root, profile, workspace, environment):
    with file_tree.open_directory(root / 'agent/config', create=True):
        pass
    directory = recording.config_directory(profile, environment)
    backend = profile['backend']
    files = []
    if backend == 'codex':
        files += [directory / 'config.toml', directory / 'AGENTS.md', directory / 'AGENTS.override.md']
        files += list((directory / 'rules').glob('*.rules'))
        files += list((directory / 'skills').glob('**/SKILL.md'))
    elif backend == 'pi':
        files += [directory / name for name in ('settings.json', 'models.json', 'SYSTEM.md', 'APPEND_SYSTEM.md')]
        files += list((directory / 'skills').glob('**/SKILL.md'))
        files += list((directory / 'extensions').glob('**/*.js')) + list((directory / 'extensions').glob('**/*.ts'))
    for parent in [*reversed(workspace.parents), workspace]:
        files += [parent / name for name in ('AGENTS.override.md', 'AGENTS.md', 'CLAUDE.md')]
        if backend in ('codex', 'pi'):
            files += list((parent / '.agents/skills').glob('**/SKILL.md'))
        if backend == 'pi':
            files += [parent / '.pi' / name for name in ('settings.json', 'SYSTEM.md', 'APPEND_SYSTEM.md')]
        if backend == 'codex':
            files += [parent / '.codex/config.toml']
    for flag in ('--system-prompt', '--append-system-prompt', '--extension', '--skill'):
        argument = recording.option(profile['argv'], flag)
        if argument:
            path = Path(argument).expanduser()
            path = path if path.is_absolute() else workspace / path
            if path.is_file():
                files.append(path)
    wrapper = common.profile_path(profile['argv'][0])
    if wrapper.name == 'launch.py':
        files.append(wrapper)
    manifest = []
    for path in dict.fromkeys(files):
        try:
            raw = recording.read_bytes(path.resolve())
        except FileNotFoundError:
            continue
        if path.suffix in ('.json', '.toml'):
            if path.suffix == '.toml':
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib
                value = tomllib.loads(raw.decode())
            else:
                value = json.loads(raw)
            content = (json.dumps(redact(value), sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
            name = path.name + '.redacted.json'
        else:
            content = redact(raw.decode()).encode()
            name = path.name
        destination = root / 'agent/config' / f'{len(manifest):04d}_{name}'
        file_tree.atomic_bytes(destination, content, mode=0o444)
        manifest.append({'source': str(path), 'path': str(destination.relative_to(root)),
                         'source_sha256': hashlib.sha256(raw).hexdigest(),
                         'saved_sha256': hashlib.sha256(content).hexdigest()})
    return {'schema_version': 1, 'backend': backend, 'config_directory': str(directory),
            'files': manifest, 'effective_settings_source': 'native session and recorded argv',
            'authentication': 'existing host credentials; not archived'}

