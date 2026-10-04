"""Probe exact provider aliases without recording credentials or raw responses."""
import concurrent.futures
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[5]
SOURCE = REPO / 'benchmarks/mig_workflow/profiles_codex/glm-5.3-flash-max'
MODELS = ('glm-5.3-flash', 'deepseek-flash', 'gpt-6.1-sol')
OUTPUT = Path(__file__).with_name('provider_probe.json')


def connection():
    config = (SOURCE / 'config.toml').read_text()
    url = json.loads(re.search(r'^base_url\s*=\s*("[^"]+")', config, re.M)[1])
    key = json.loads((SOURCE / 'auth.json').read_text())['OPENAI_API_KEY']
    if not key or not url.startswith('https://'):
        raise ValueError('provider configuration is unavailable')
    return url.rstrip('/'), key


def probe(model, url, key):
    payload = {'model': model, 'input': 'Reply with exactly READY.',
               'max_output_tokens': 1024, 'stream': False, 'store': False}
    request = urllib.request.Request(url + '/responses',
        data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    started = time.monotonic()
    row = {'requested_model': model, 'protocol': 'responses'}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=120) as response:
            row['http_status'] = response.status
            result = json.load(response)
        texts = [part.get('text', '') for item in result.get('output', [])
                 for part in item.get('content', []) if part.get('type') == 'output_text']
        row.update(returned_model=result.get('model'), response_status=result.get('status'),
                   ready_received='READY' in ''.join(texts),
                   output_types=[item.get('type') for item in result.get('output', [])])
        row['status'] = 'PASS' if row['ready_received'] else 'FAIL'
    except urllib.error.HTTPError as error:
        row.update(http_status=error.code, status='FAIL', error_type='HTTPError')
        # Only safe structured error codes are retained, never raw server output.
        try:
            value = json.load(error).get('error', {})
            for name in ('type', 'code'):
                code = value.get(name)
                if isinstance(code, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', code):
                    row['error_' + name] = code
        except Exception:
            pass
    except Exception as error:
        row.update(status='FAIL', error_type=type(error).__name__)
    row['elapsed_seconds'] = round(time.monotonic() - started, 3)
    print(json.dumps(row), flush=True)
    return row


if __name__ == '__main__':
    endpoint, credential = connection()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda model: probe(model, endpoint, credential), MODELS))
    OUTPUT.write_text(json.dumps({'source_profile': str(SOURCE), 'base_url': endpoint,
        'credential_source': 'existing profile auth.json; value not recorded',
        'transport': 'direct; this endpoint times out through open_proxy',
        'models': rows}, indent=2) + '\n')
