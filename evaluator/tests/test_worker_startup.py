"""Actual isolated Python children share the trusted ISA without SDK writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import isolation
import environment_lock
from worker_startup import sitecustomize_source


def test_trusted_worker_launcher_keeps_broker_host_identity():
    environment = isolation.clean_environment({})
    process = subprocess.Popen(
        isolation.View().system().argv(['/usr/bin/sleep', '5'], cwd='/tmp', environment=environment),
        cwd='/', env=isolation.sandbox_launcher_environment(environment),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        status_path = Path(f'/proc/{process.pid}/status')
        deadline = time.monotonic() + 5
        while not status_path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert process.poll() is None, process.stderr.read().decode()
        fields = {
            line.partition(':')[0]: line.partition(':')[2].strip()
            for line in status_path.read_text().splitlines()
            if ':' in line
        }
        assert [int(value) for value in fields['Uid'].split()] == [os.getuid()] * 4
        assert [int(value) for value in fields['Gid'].split()] == [os.getgid()] * 4
        assert [int(value) for value in fields['Groups'].split()] == sorted(os.getgroups())
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=5)


@pytest.mark.parametrize('bad_pin', [False, True])
def test_verified_isa_reaches_python_children_in_readonly_namespace(tmp_path, bad_pin):
    from pypto.runtime.pto_isa import ensure_pto_isa_root
    import simpler_setup.pto_isa as resolver
    root = Path(ensure_pto_isa_root())
    revision = resolver.read_pto_isa_pin()
    case_site_packages = Path(
        json.loads(environment_lock.lock_path().read_text())['runtime']['installed_assets']
    ).parents[1]
    metadata = tmp_path/'worker.json'
    metadata.write_text(json.dumps({'pto_isa_root': str(root),
        'pto_isa_revision': '0'*40 if bad_pin else revision}))
    startup = tmp_path/'startup'
    startup.mkdir()
    (startup/'sitecustomize.py').write_text(sitecustomize_source())
    (startup/'worker_startup.py').write_bytes((Path(__file__).parents[1]/'worker_startup.py').read_bytes())
    view = (isolation.View().system().add(metadata, '/ptobench/worker.json')
            .add(startup, '/ptobench/python_startup').add(case_site_packages))
    for headers in ('include', 'pkg_inc'):
        if (root/headers).is_dir():
            view.add(root/headers)
    env = isolation.clean_environment({**os.environ,
        'PYTHONPATH': '/ptobench/python_startup:' + str(case_site_packages)})
    probe = ('import pathlib; import pypto.runtime.pto_isa as p; '
             'import simpler_setup.pto_isa as r; '
             f'assert p.ensure_pto_isa_root()==r.ensure_pto_isa_root()=={str(root)!r}; '
             f'assert not pathlib.Path({str(root / ".git")!r}).exists(); '
             'print("CHILD_PIN_OK")')
    python = os.path.abspath(sys.executable)
    runtime_environment = repr({'PYTHONPATH': '/public/case:' + str(case_site_packages)})
    parent = ('import subprocess,sys,os; from worker_startup import runtime_environment; '
              'os.environ.update(runtime_environment(' + runtime_environment + '))\n'
              + probe + '\nsubprocess.run([sys.executable,"-c",' + repr(probe) + '],check=True)')
    result = subprocess.run(view.argv([python, '-c', parent], cwd='/tmp', environment=env),
                            env=isolation.sandbox_launcher_environment(env),
                            capture_output=True, text=True, timeout=90)
    foreign = subprocess.run(view.argv(['/usr/bin/python3', '-c', 'print(\"FOREIGN_OK\")'],
        cwd='/tmp', environment=env), env=isolation.sandbox_launcher_environment(env),
        capture_output=True, text=True, timeout=20)
    assert foreign.returncode == 0, foreign.stderr
    assert 'FOREIGN_OK' in foreign.stdout
    if bad_pin:
        assert result.returncode == 70
        assert 'revision differs' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('CHILD_PIN_OK') == 2
