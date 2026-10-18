"""Collect raw Case environment metadata, independent of optimizer methods."""
import subprocess
import sys

import common
import session

def collect_environment(root, case_id):
    """Probe the Case interpreter after its normal bootstrap; never execute a Case."""
    path = root / 'environment_probe.json'
    completed = subprocess.run([sys.executable, str(common.BENCHMARK_ROOT / 'evaluator/inspect_environment.py'),
                               '--case', case_id, '--result', str(path)],
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if completed.returncode != 0 or not path.is_file():
        raise RuntimeError('Case environment provenance probe failed')
    return session._read(path)
