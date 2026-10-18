"""Strict experiment YAML; participant implementations own backend settings."""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import re

import yaml
import common
import file_tree
import session
from launcher.methods import load_method

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


@dataclass
class Experiment:
    case: str
    optimizer: dict
    output: dict
    budget: dict
    source: str
    mode: str = 'basic'

    def to_dict(self):
        return {'case': self.case, 'mode': self.mode, 'optimizer': self.optimizer, 'output': self.output,
                'budget': self.budget, 'source': self.source}


def load_experiment(path):
    path = file_tree.absolute_path(path)
    value = mapping(read_yaml(path), ('case', 'mode', 'optimizer', 'output', 'budget'), ('case', 'optimizer'))
    mode = value.get('mode', 'basic')
    try:
        load_method(mode)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    case = identifier(value['case'], 'Case')
    if len(case.split('.')) != 2:
        raise ConfigError('Case must be family.name')
    optimizer = dict(mapping(value['optimizer'], ('id', 'backend', 'profile'), ('id', 'backend')))
    for key in ('id', 'backend'):
        identifier(optimizer[key], key)
    optimizer['profile'] = identifier(optimizer.get('profile', 'default'), 'profile')
    output = mapping(value.get('output', {}), ('runs_root', 'snapshots_root'))
    normalized = {}
    for name, default in (('runs_root', 'runs'), ('snapshots_root', 'snapshots')):
        target = output.get(name, default)
        if not isinstance(target, str) or not target:
            raise ConfigError('output paths must be nonempty strings')
        normalized[name] = str(path.parent / target) if not Path(target).is_absolute() else target
    runs, snapshots = session._output(normalized)
    budget = session._budget(value.get('budget'))
    return Experiment(case, optimizer, {'runs_root': str(runs), 'snapshots_root': str(snapshots)}, budget, str(path), mode)
