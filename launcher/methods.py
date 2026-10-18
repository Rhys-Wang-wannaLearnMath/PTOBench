"""Opaque participant entrypoints: no agent or optimization policy belongs here."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import common
import file_tree


class MethodError(ValueError):
    def __init__(self, message, error_code='invalid_request'):
        super().__init__(message)
        self.error_code = error_code


def load_method(name):
    import yaml
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise MethodError('invalid experiment mode')
    directory = common.BENCHMARK_ROOT / 'modes' / name
    try:
        with file_tree.open_file(directory / 'mode.yaml') as fd, os.fdopen(os.dup(fd), 'rb') as stream:
            raw = stream.read()
        value = yaml.safe_load(raw)
        if (not isinstance(value, dict) or set(value) != {'schema_version', 'id', 'implemented', 'description', 'entrypoint'}
                or type(value['schema_version']) is not int or value['schema_version'] != 2
                or value['id'] != name or type(value['implemented']) is not bool
                or not isinstance(value['description'], str)):
            raise MethodError('invalid experiment method definition')
        if not value['implemented']:
            raise MethodError(f'experiment mode {name} is not implemented yet')
        entry = value['entrypoint']
        if not isinstance(entry, str) or not re.fullmatch(r'[A-Za-z0-9_/-]+\.py', entry) or '..' in entry or entry.startswith('/'):
            raise MethodError('invalid method entrypoint')
        with file_tree.open_file(directory / entry) as fd, os.fdopen(os.dup(fd), 'rb') as stream:
            source = stream.read()
        return {**value, 'path': str(directory / entry), 'files': {
            'mode.yaml': hashlib.sha256(raw).hexdigest(), entry: hashlib.sha256(source).hexdigest()}}
    except (OSError, yaml.YAMLError, UnicodeError, file_tree.UnsafeTreeError) as error:
        raise MethodError(f'experiment mode {name} is unavailable') from error


def command(method, operation, request):
    return [sys.executable, method['path'], operation, str(request)]


def call(method, operation, payload, *, environment=None):
    """Invoke a bounded metadata operation; run itself has the normal lifecycle."""
    env = {**(os.environ if environment is None else environment), 'PYTHONDONTWRITEBYTECODE': '1'}
    with tempfile.TemporaryDirectory(prefix='ptobench_method_') as temporary:
        request = Path(temporary) / 'request.json'
        request.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        try:
            completed = subprocess.run(command(method, operation, request), env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            value = json.loads(completed.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            raise MethodError(f'method {operation} did not return a valid response', 'infrastructure_error') from error
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise MethodError('invalid method response', 'infrastructure_error')
    if completed.returncode != 0:
        code = value.get('error_code')
        # Entry points must return non-sensitive protocol errors, not terminal dumps.
        raise MethodError(value.get('message', 'method operation failed'),
                          code if code in ('invalid_request', 'infrastructure_error') else 'infrastructure_error')
    return value


def start(method, request, workspace, environment):
    """Launch any participant in an owned group; Basic may exec its native tool."""
    import session
    process = subprocess.Popen(command(method, 'run', request), cwd=workspace,
                               env={**environment, 'PYTHONDONTWRITEBYTECODE': '1'}, shell=False,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    process.ptobench_identity = session.process_identity(process.pid)
    if process.ptobench_identity is None:
        data = Path(f'/proc/{process.pid}/stat').read_text()
        fields = data[data.rfind(')') + 2:].split()
        process.ptobench_identity = {'pid': process.pid, 'start_ticks': int(fields[19]), 'pgid': int(fields[2])}
    return process
