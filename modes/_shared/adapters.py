"""Four thin adapters sharing one direct subprocess runner."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import os
import shutil
import subprocess
from pathlib import Path

from modes._shared.config import BACKENDS, ConfigError, render


class BackendUnavailable(RuntimeError):
    error_code = 'infrastructure_error'


@dataclass
class BackendInfo:
    backend: str
    executable: str
    version: str
    model: str
    authentication: str
    checked_help: bool

    def to_dict(self):
        return asdict(self)


class Adapter:
    def __init__(self, backend):
        if backend not in BACKENDS:
            raise ConfigError('undefined backend')
        self.backend = backend

    def preflight(self, profile):
        if profile['backend'] != self.backend:
            raise ConfigError('profile backend mismatch')
        environment = {**os.environ, **profile['env']}
        # Profile argv may contain the portable {repo_root} placeholder.
        # Resolve it before checking the executable, while keeping the
        # recorded profile independent of the checkout location.
        argv, _ = render(profile, Path.cwd(), Path.cwd(), '')
        executable = shutil.which(argv[0], path=environment.get('PATH'))
        if executable is None:
            raise BackendUnavailable(f'{self.backend} executable is not installed')
        def query(argv):
            try:
                completed = subprocess.run(argv, shell=False, env=environment, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise BackendUnavailable(f'{self.backend} tool query failed') from error
            if completed.returncode != 0:
                raise BackendUnavailable(f'{self.backend} tool query returned nonzero')
            return completed.stdout.decode(errors='replace').strip()
        version = query(profile['version_argv'])
        if not version:
            raise BackendUnavailable(f'{self.backend} version is unavailable')
        checked = False
        if 'help_argv' in profile:
            help_text = query(profile['help_argv'])
            for flag in argv[1:]:
                if flag.startswith('--') and '=' not in flag and flag not in help_text:
                    raise BackendUnavailable(f'{self.backend} installed version does not support a configured flag')
            checked = True
        authentication = 'existing_host_configuration'
        if self.backend == 'codex' and os.path.basename(profile['argv'][0]) == 'codex':
            try:
                auth = subprocess.run([executable, 'login', 'status'], shell=False, env=environment,
                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise BackendUnavailable('codex authentication status unavailable') from error
            if auth.returncode != 0:
                raise BackendUnavailable('codex has no existing authentication')
            # No credential or original authentication output is recorded.
            authentication = 'authenticated_host_configuration'
        if self.backend == 'claudecode' and os.path.basename(profile['argv'][0]) == 'claude':
            try:
                auth = subprocess.run([executable, 'auth', 'status'], shell=False, env=environment,
                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                authenticated = auth.returncode == 0 and json.loads(auth.stdout).get('loggedIn') is True
            except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                raise BackendUnavailable('claudecode authentication status unavailable') from error
            if not authenticated:
                raise BackendUnavailable('claudecode has no existing authentication')
            authentication = 'authenticated_host_configuration'
        from modes._shared.recording import codex_override, option
        override = codex_override(argv, 'model') if self.backend == 'codex' else None
        model = option(argv, '--model') or option(argv, '-m') or override or profile.get('model', 'unknown')
        provider = option(argv, '--provider')
        if provider and '/' not in model:
            model = provider + '/' + model
        return BackendInfo(self.backend, executable, version, model, authentication, checked)



class CodexAdapter(Adapter):
    def __init__(self): super().__init__('codex')


class ClaudeCodeAdapter(Adapter):
    def __init__(self): super().__init__('claudecode')


class PiAdapter(Adapter):
    def __init__(self): super().__init__('pi')


class OpenCodeAdapter(Adapter):
    def __init__(self): super().__init__('opencode')


def get_adapter(backend):
    return {'codex': CodexAdapter, 'claudecode': ClaudeCodeAdapter, 'pi': PiAdapter, 'opencode': OpenCodeAdapter}[backend]()
