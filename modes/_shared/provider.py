"""Optional mode-local Pi provider settings; credentials stay out of Run records."""
import hashlib
import json
import os
from pathlib import Path
import shlex
from urllib.parse import urlsplit

import common
import file_tree
from modes._shared.config import ConfigError, identifier
from modes._shared import recording


FIELDS = {'LLM_PROVIDER', 'LLM_BASE_URL', 'LLM_API', 'LLM_API_KEY',
          'LLM_MODEL', 'LLM_REASONING'}
LOCAL_GATEWAY_URL = 'http://127.0.0.1:18081'


def read_env(path):
    """Read literal assignments, without shell execution or variable expansion."""
    try:
        values = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].lstrip()
            key, separator, value = line.partition('=')
            key = key.strip()
            if not separator or key not in FIELDS or key in values:
                raise ValueError()
            parts = shlex.split(value, comments=True)
            if len(parts) > 1 or '\0' in value:
                raise ValueError()
            values[key] = parts[0] if parts else ''
        return values
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError('cannot read mode .env; use literal LLM_*=value assignments') from error


def api_key(profile):
    config = profile.get('mode_provider')
    if not config:
        return None
    key = read_env(config['env_file']).get('LLM_API_KEY') or os.environ.get('PYPTO_PROVIDER_API_KEY')
    if not key:
        source = common.REPO_ROOT / 'benchmarks/mig_workflow/profiles_codex/glm-5.3-flash-max/auth.json'
        try:
            key = json.loads(source.read_text()).get('OPENAI_API_KEY')
        except (OSError, ValueError):
            pass
    if not isinstance(key, str) or not key:
        raise ConfigError('set LLM_API_KEY in the mode .env or PYPTO_PROVIDER_API_KEY in the environment')
    return key


def credentials(profile):
    if os.environ.get('PTOBENCH_PROVIDER_PREPARED') == '1':
        key = os.environ.get('PYPTO_PROVIDER_API_KEY')
        return {'PYPTO_PROVIDER_API_KEY': key} if key else {}
    key = api_key(profile)
    return {'PYPTO_PROVIDER_API_KEY': key} if key else {}


def local_gateway_url(profile):
    config = profile.get('mode_provider')
    if not config:
        raise ConfigError('local model gateway requires mode-local provider settings')
    path = urlsplit(config['base_url']).path.rstrip('/')
    return LOCAL_GATEWAY_URL + path


def configure(profile, mode_directory):
    path = Path(mode_directory).resolve() / '.env'
    if profile['backend'] != 'pi':
        return profile
    limits_requested = any(key in profile for key in ('context_window', 'max_output_tokens'))
    if not path.is_file():
        if limits_requested:
            raise ConfigError('Pi profile token limits require a mode-local provider .env')
        return profile
    values = read_env(path)
    if not values:
        if limits_requested:
            raise ConfigError('Pi profile token limits require provider settings in the mode .env')
        return profile
    base_url, model = values.get('LLM_BASE_URL', ''), values.get('LLM_MODEL', '')
    name = values.get('LLM_PROVIDER') or 'provider-default'
    reasoning = values.get('LLM_REASONING') or profile.get('reasoning', 'off')
    try:
        url = urlsplit(base_url)
        valid_url = url.scheme in ('http', 'https') and url.hostname and not url.username and not url.password
    except ValueError:
        valid_url = False
    if not valid_url or not model or model.startswith('-'):
        raise ConfigError('mode .env requires LLM_BASE_URL (http/https) and a selected model')
    identifier(name, 'mode provider')
    api = values.get('LLM_API') or 'openai-responses'
    if reasoning not in ('off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
        raise ConfigError('invalid mode .env reasoning')
    # Preserve existing model limits and compatibility, without copying keys.
    model_config = {'id': model}
    try:
        existing = json.loads((recording.config_directory(profile) / 'models.json').read_text())
        for item in existing.get('providers', {}).values():
            for registered in item.get('models', []):
                if registered.get('id') == model:
                    model_config.update({key: registered[key] for key in
                        ('name', 'input', 'contextWindow', 'maxTokens', 'reasoning', 'thinkingLevelMap', 'compat', 'cost') if key in registered})
    except (OSError, ValueError):
        pass
    if 'context_window' in profile:
        model_config['contextWindow'] = profile['context_window']
    if 'max_output_tokens' in profile:
        limit = profile['max_output_tokens']
        if limit == 'unlimited':
            if api == 'openai-responses':
                model_config.pop('maxTokens', None)
            elif api == 'openai-completions':
                # Pi requires positive model metadata; a request hook removes
                # its otherwise automatic Completions output cap in prepare().
                model_config.pop('maxTokens', None)
            else:
                raise ConfigError('unlimited Pi output requires openai-responses or openai-completions')
        else:
            model_config['maxTokens'] = limit
        if api == 'openai-responses':
            model_config['compat'] = {**model_config.get('compat', {}),
                                      'supportsMaxOutputTokens': limit != 'unlimited'}
    argv = list(profile['argv'])
    if argv[:2] == ['{repo_root}/benchmarks/mig_workflow/agent_provider/launch.py', 'pi']:
        argv = argv[1:]
    selected = []
    index = 0
    while index < len(argv):
        if argv[index].split('=', 1)[0] in ('--provider', '--model', '-m', '--thinking'):
            index += 1 if '=' in argv[index] else 2
        else:
            selected.append(argv[index])
            index += 1
    index = selected.index('{prompt}')
    selected[index:index] = ['--provider', name, '--model', model, '--thinking', reasoning]
    result = {**profile, 'argv': selected, 'model': name + '/' + model, 'reasoning': reasoning,
              'mode_provider': {'env_file': str(path), 'provider': name, 'base_url': base_url,
                                'api': api, 'model': model_config}}
    credentials(result)  # Validate before creating a Run; never return the key.
    result['sha256'] = hashlib.sha256(json.dumps(
        {key: value for key, value in result.items() if key != 'sha256'}, sort_keys=True).encode()).hexdigest()
    return result


def prepare(profile, run_root, *, gateway=False):
    config = profile.get('mode_provider')
    if not config:
        return profile
    directory = Path(run_root) / 'agent/pi'
    with file_tree.open_directory(directory, create=True) as descriptor:
        os.fchmod(descriptor, 0o700)
    settings = {'defaultProvider': config['provider'], 'defaultModel': config['model']['id'],
                'defaultThinkingLevel': profile['reasoning']}
    base_url = local_gateway_url(profile) if gateway else config['base_url']
    models = {'providers': {config['provider']: {'baseUrl': base_url, 'api': config['api'],
              'apiKey': '$PYPTO_PROVIDER_API_KEY', 'models': [config['model']]}}}
    for name, value in (('settings.json', settings), ('models.json', models)):
        file_tree.atomic_bytes(directory / name, (json.dumps(value, indent=2) + '\n').encode(), mode=0o600)
    argv = profile['argv']
    if config['api'] == 'openai-completions' and profile.get('max_output_tokens') == 'unlimited':
        extension = directory / 'unlimited-output.ts'
        file_tree.atomic_bytes(extension, (
            'export default function(pi) {\n'
            '  pi.on("before_provider_request", (event) => {\n'
            '    const payload = { ...event.payload };\n'
            '    delete payload.max_tokens;\n'
            '    delete payload.max_completion_tokens;\n'
            '    return payload;\n'
            '  });\n'
            '}\n').encode(), mode=0o600)
        argv = list(argv)
        argv[argv.index('{prompt}'):argv.index('{prompt}')] = ['--extension', str(extension)]
    return {**profile, 'argv': argv, 'env': {**profile['env'], 'PI_CODING_AGENT_DIR': str(directory), 'PI_OFFLINE': '1'},
            'recording': {**profile.get('recording', {}), 'config_directory': str(directory)}}
