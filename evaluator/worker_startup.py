"""Read-only compiler identity shared by an isolated worker and its children.

The host validates the ISA checkout before exposing only its headers. Every
Python subprocess needs the same resolver, without touching a shared SDK lock
or falling back to mutable ambient PTO_ISA_ROOT values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def install(metadata_path='/ptobench/worker.json'):
    metadata = json.loads(Path(metadata_path).read_text())
    root = metadata.get('pto_isa_root')
    if root is None:
        return
    path = Path(root)
    revision = metadata.get('pto_isa_revision')
    if not path.is_absolute() or not (path / 'include').is_dir():
        raise RuntimeError('verified worker ISA headers are unavailable')
    import simpler_setup.pto_isa as runtime_isa
    if not revision or runtime_isa.read_pto_isa_pin() != revision:
        raise RuntimeError('verified worker ISA revision differs from installed pin')
    import pypto.runtime.pto_isa as isa
    isa._resolve_pinned_pto_isa_root = lambda: str(path)
    runtime_isa.ensure_pto_isa_root = lambda **kwargs: str(path)
    import pypto.runtime.device_runner as device_runner
    original = device_runner._clean_git_revision
    if getattr(original, '_ptobench_verified_isa', False):
        return
    def clean_revision(candidate):
        if Path(candidate) == path:
            return True, revision
        return original(candidate)
    clean_revision._ptobench_verified_isa = True
    device_runner._clean_git_revision = clean_revision


def sitecustomize_source():
    # sitecustomize exceptions are otherwise printed and ignored by Python.
    # Fail closed so a bad pin cannot silently fall back to another resolver.
    return "import os, sys\nif os.path.realpath(sys.prefix) == " + repr(str(Path(sys.prefix).resolve())) + ":\n" + """    try:
        from worker_startup import install
        install('/ptobench/worker.json')
    except BaseException as error:
        sys.stderr.write('PTOBench worker startup failed: ' + str(error) + '\\n')
        sys.stderr.flush()
        os._exit(70)
"""


def runtime_environment(updates):
    """Retain the trusted hook when a Case overrides PYTHONPATH for execution."""
    result = dict(updates)
    startup = Path('/ptobench/python_startup')
    if (startup / 'sitecustomize.py').is_file():
        result['PYTHONPATH'] = str(startup) + ':' + str(Path(__file__).parent) + ':' + result.get('PYTHONPATH', '')
    return result
