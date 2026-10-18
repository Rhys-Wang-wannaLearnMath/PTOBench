"""Strict safe YAML and named, non-sensitive tool profiles."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re

import yaml
import common
import file_tree

BACKENDS = ('codex', 'claudecode', 'pi', 'opencode')
PLACEHOLDERS = {'prompt', 'workspace', 'session_root', 'repo_root'}
SENSITIVE = re.compile(r'(?:api.?key|token|secret|password|credential|authorization)', re.I)


class ConfigError(ValueError):
    error_code = 'invalid_request'


def mapping(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or not set(required) <= value.keys():
        raise ConfigError('invalid configuration fields or missing required field')
    return value


def identifier(value, label):
    if (not isinstance(value, str) or '..' in value
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value)):
        raise ConfigError(f'invalid {label}')
    return value


def read_yaml(path):
    try:
        with file_tree.open_file(path) as fd, os.fdopen(os.dup(fd), 'r') as reader:
            if os.fstat(fd).st_size > 1024 * 1024:
                raise ConfigError('configuration is too large')
            source = reader.read()
            value = yaml.safe_load(source)
            # SafeLoader accepts duplicate keys; strict configuration must not.
            visited = set()
            def check(node):
                if node is None or id(node) in visited:
                    return
                visited.add(id(node))
                if isinstance(node, yaml.MappingNode):
                    keys = []
                    for key, child in node.value:
                        if not isinstance(key, yaml.ScalarNode) or key.value in keys:
                            raise ConfigError('duplicate or non-scalar YAML key')
                        keys.append(key.value)
                        check(child)
                elif isinstance(node, yaml.SequenceNode):
                    for child in node.value:
                        check(child)
            check(yaml.compose(source, Loader=yaml.SafeLoader))
        return mapping(value, value.keys() if isinstance(value, dict) else ())
    except (OSError, yaml.YAMLError, file_tree.UnsafeTreeError, UnicodeError) as error:
        # YAML errors include source lines; never echo potential credentials.
        raise ConfigError('cannot read safe YAML mapping') from error


def _argv(value, label):
    if not isinstance(value, list) or not value or any(not isinstance(arg, str) or '\0' in arg for arg in value) or not value[0]:
        raise ConfigError(f'{label} must be a nonempty string argv array')
    if any(SENSITIVE.search(arg) for arg in value if arg.startswith('-')):
        raise ConfigError('credential arguments must use existing host authentication')
    for arg in value:
        if set(re.findall(r'\{([A-Za-z_][A-Za-z0-9_]*)\}', arg)) - PLACEHOLDERS:
            raise ConfigError('unknown profile placeholder')
    return value


def load_profile(backend, name='default', root=None):
    identifier(backend, 'backend'); identifier(name, 'profile')
    if backend not in BACKENDS:
        raise ConfigError('undefined backend')
    if root is None:
        # Read existing custom profiles at their old location without moving policy
        # back into the evaluator or the generic Launcher.
        legacy = common.BENCHMARK_ROOT / 'launcher/profiles'
        root = legacy if (legacy / backend / f'{name}.yaml').is_file() else common.BENCHMARK_ROOT / 'modes/_shared/profiles'
    root = Path(root)
    path = root / backend / f'{name}.yaml'
    value = dict(mapping(read_yaml(path), ('argv', 'version_argv', 'help_argv', 'env', 'model', 'reasoning'), ('argv', 'version_argv', 'env')))
    # The monorepo profiles may use its provider launcher.  A standalone
    # PTOBench checkout does not contain that integration layer; in that case
    # run the backend directly while preserving the nested behavior.
    argv = _argv(value['argv'], 'argv')
    provider_launcher = '{repo_root}/benchmarks/mig_workflow/agent_provider/launch.py'
    if backend in ('pi', 'opencode') and argv[:2] == [provider_launcher, backend]:
        if not common.profile_path(argv[0]).is_file():
            value['argv'] = argv[1:]
            if '--model' not in value['argv']:
                value.pop('model', None)
    for key in ('argv', 'version_argv'):
        _argv(value[key], key)
    if value['argv'].count('{prompt}') != 1 or any('{prompt}' in arg and arg != '{prompt}' for arg in value['argv']):
        raise ConfigError('prompt must be one separate argv element')
    if any('{' in arg or '}' in arg for arg in value['version_argv']):
        raise ConfigError('version argv cannot contain placeholders')
    if 'help_argv' in value:
        _argv(value['help_argv'], 'help_argv')
        if any('{' in arg or '}' in arg for arg in value['help_argv']):
            raise ConfigError('help argv cannot contain placeholders')
    env = value['env']
    if (not isinstance(env, dict) or any(not isinstance(k, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k)
            or not isinstance(v, str) or '\0' in v or SENSITIVE.search(k)
            or k == 'PATH' or k.startswith('PTOBENCH_') for k, v in env.items())):
        raise ConfigError('profile env accepts only non-sensitive string settings')
    # Reject accidental credentials embedded in config JSON or argv text too.
    if any(re.search(r'(?:sk-[A-Za-z0-9_-]{10,}|Bearer\s+\S+|["\'](?:api_key|token|password)["\']\s*:)', arg, re.I)
           for arg in [*value['argv'], *env.values()]):
        raise ConfigError('profile contains sensitive configuration')
    if 'model' in value and (not isinstance(value['model'], str) or not value['model']):
        raise ConfigError('model must be text')
    if 'reasoning' in value and (not isinstance(value['reasoning'], str) or not value['reasoning']):
        raise ConfigError('reasoning must be text')
    encoded = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    return {**value, 'backend': backend, 'name': name, 'sha256': hashlib.sha256(encoded).hexdigest(), 'path': str(path)}


def render(profile, workspace, session_root, prompt):
    values = {'workspace': str(workspace), 'session_root': str(session_root),
              'prompt': prompt, 'repo_root': str(common.REPO_ROOT)}
    def substitute(text):
        return re.sub(r'\{(prompt|workspace|session_root|repo_root)\}', lambda m: values[m[1]], text)
    return [substitute(arg) for arg in profile['argv']], {key: substitute(value) for key, value in profile['env'].items()}
