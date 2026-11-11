"""Read-only environment collection used to bind cache and applicability."""
from __future__ import annotations

from pathlib import Path
import importlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
import hashlib
import shutil
import os
import re
import configparser

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import BuilderConfig


def _run(argv: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)
        return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:]}
    except Exception as exc:
        return {"argv": argv, "error": type(exc).__name__}


def _acl_runtime_probe(python: str) -> dict:
    """Read ACL runtime/device count in a disposable process without allocating a device."""
    code = (
        "import acl,json\n"
        "out={}\n"
        "try:\n"
        " out['init']=acl.init(); out['device_count']=acl.rt.get_device_count(); out['run_mode']=acl.rt.get_run_mode()\n"
        "except Exception as e: out['error']=type(e).__name__\n"
        "finally:\n"
        " try: out['finalize']=acl.finalize()\n"
        " except Exception as e: out['finalize_error']=type(e).__name__\n"
        "print(json.dumps(out,sort_keys=True))\n"
    )
    result = _run([python, '-c', code], timeout=20)
    if result.get('returncode') == 0:
        try: result['value'] = json.loads(result.get('stdout', ''))
        except Exception: result['value'] = None
    return result


def _backend_budget(skus):
    """Query installed compiler registry; never label its budget physical capacity."""
    if not skus or any(not sku.startswith('910B') for sku in skus):
        return {'value': None, 'reason': 'reported hardware does not identify a supported compiler target'}
    try:
        from pypto import backend, ir
        target = backend.get_backend_instance(backend.BackendType.Ascend910B)
        spaces = {name: target.get_mem_size(getattr(ir.MemorySpace, name))
                  for name in ('Vec', 'Mat', 'Left', 'Right', 'Acc', 'Bias')}
        if any(type(value) is not int or value <= 0 for value in spaces.values()):
            raise ValueError('invalid compiler budget')
        return {'value': spaces, 'unit': 'byte', 'target_backend': target.get_type_name(),
                'source': 'installed public Backend.get_mem_size registry',
                'scope': 'compiler configured allocation budget; not physical capacity or current allocation'}
    except Exception as exc:
        return {'value': None, 'reason': 'compiler budget query unavailable', 'error': type(exc).__name__}


def _hardware_definition(skus, cann_root):
    result = {'verified': False, 'reported_skus': skus, 'physical_capacities_bytes': None,
              'reason': 'no exact installed SDK hardware definition matched to reported SKU'}
    if len(skus) != 1 or not cann_root or not re.fullmatch(r'910B[0-9]*(?:-[0-9]+)?', skus[0]):
        return result
    path = Path(cann_root) / 'aarch64-linux/data/platform_config' / ('Ascend' + skus[0] + '.ini')
    try:
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        parser.read_string(path.read_text())
        if parser['version']['SoC_version'] != 'Ascend' + skus[0]:
            return result
        capacities = {label: parser.getint(section, key) for label, section, key in (
            ('UB', 'AICoreSpec', 'ub_size'), ('L1', 'AICoreSpec', 'l1_size'),
            ('L0A', 'AICoreSpec', 'l0_a_size'), ('L0B', 'AICoreSpec', 'l0_b_size'),
            ('L0C', 'AICoreSpec', 'l0_c_size'), ('L2', 'SoCInfo', 'l2_size'), ('HBM', 'SoCInfo', 'memory_size'))}
        cores = {label: parser.getint('SoCInfo', key) for label, key in (('Cube', 'cube_core_cnt'), ('Vector', 'vector_core_cnt'))}
        if any(value <= 0 for value in list(capacities.values()) + list(cores.values())):
            return result
        return {'verified': True, 'reported_skus': skus, 'architecture': parser['version']['AIC_version'],
                'physical_capacities_bytes': capacities, 'core_counts': cores,
                'definition_file': str(path), 'definition_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'scope': 'installed SDK hardware definition matched to read-only reported SKU; not current allocation or a device measurement'}
    except (OSError, ValueError, KeyError, configparser.Error):
        return result


def _collect_current() -> dict:
    py = sys.executable
    packages = {}
    for name in ("pypto", "torch", "torch_npu", "acl", "simpler_setup", "simpler"):
        try:
            mod = importlib.import_module(name)
            packages[name] = {"version": getattr(mod, "__version__", None), "file": str(getattr(mod, "__file__", ""))}
            if name == 'pypto':
                core = importlib.import_module('pypto.pypto_core')
                core_path = Path(core.__file__)
                with core_path.open('rb') as stream:
                    checksum = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        checksum.update(block)
                packages[name]['core_implementation_sha256'] = checksum.hexdigest()
        except Exception as exc:
            try: version = importlib.metadata.version(name)
            except Exception: version = None
            packages[name] = {"version": version, "error": type(exc).__name__}
    api = {}
    try:
        mod = importlib.import_module("pypto.language")
        for name in ("function", "program", "jit", "unroll", "Tensor", "Tile", "In", "Out", "InOut", "reshape", "transpose", "cast", "matmul", "matmul_acc", "load", "store", "range", "parallel", "pipeline", "at", "scope", "incore", "outcore", "create_tensor", "create_tile", "assemble", "slice", "full", "zeros", "MemRef", "set_validshape", "fillpad", "PadValue"):
            obj = getattr(mod, name, None)
            try: signature = str(inspect.signature(obj)) if obj is not None else None
            except (TypeError, ValueError): signature = None
            try:
                implementation = inspect.getsourcefile(obj)
                digest = hashlib.sha256(Path(implementation).read_bytes()).hexdigest() if implementation else None
            except (TypeError, OSError): digest = None
            api[name] = {"present": obj is not None, "signature": signature, "implementation_sha256": digest}
    except Exception as exc:
        api["error"] = type(exc).__name__
    device_help = _run(["npu-smi", "info", "-h"], timeout=5)
    device_probe = _run(["npu-smi", "info"], timeout=10) if device_help.get('returncode') == 0 else {'error': 'npu-smi info help unavailable'}
    device_text=device_probe.get("stdout","")
    device_present=device_probe.get("returncode") == 0 and "910B" in device_text
    reported_skus = sorted(set(re.findall(r'\b(?:910B\w*(?:-\w+)*|950\w*(?:-\w+)*)\b', device_text)))
    device_ready=device_present and packages.get("torch_npu",{}).get("file") is not None
    acl_probe = _acl_runtime_probe(py)
    executables = {}
    for name in ("ptoas", "bisheng", "ccec"):
        location = shutil.which(name) or (str(Path(sys.executable).parent / name) if (Path(sys.executable).parent / name).is_file() else None)
        executables[name] = {"path": location, "version": _run([location, "--version"], timeout=5) if location else None}
    cann = {"root": os.environ.get("ASCEND_HOME_PATH"), "version": None}
    candidates = [Path(cann['root']) / 'version.cfg'] if cann['root'] else []
    if cann['root']:
        candidates += [Path(cann['root']) / 'aarch64-linux/ascend_toolkit_install.info']
    candidates += [Path('/usr/local/Ascend/ascend-toolkit/latest/version.cfg')]
    for candidate in candidates:
        if candidate.is_file():
            cann['version'] = candidate.read_text(errors='replace')
            cann['metadata_path'] = str(candidate)
            break

    return {"python": {"executable": py, "version": sys.version, "platform": platform.platform()}, "packages": packages, "api": api, "backend_budget": _backend_budget(reported_skus), "hardware": _hardware_definition(reported_skus, cann['root']), "toolchain": {"executables": executables, "cann": cann, "pi": _run(["pi", "--version"]), "pi_help": _run(["pi", "--help"])}, "device": {"present": device_present, "available": device_ready, 'reported_skus': reported_skus, 'assigned': False, "acl_runtime": acl_probe.get('value'), "reason": "torch_npu import unavailable; ACL count is read-only and no device allocation was performed" if device_present and not device_ready else ("device probe unavailable" if not device_present else "ready; explicit allocation still required")}}


def environment_identity(environment):
    return {key: environment.get(key, {}) for key in ('python', 'packages', 'api')} | {
        'toolchain': {key: value for key, value in environment.get('toolchain', {}).items() if key != 'pi_help'},
        'backend_budget': environment.get('backend_budget', {}),
        'hardware': environment.get('hardware', {}),
        'device_skus': environment.get('device', {}).get('reported_skus', [])}


def release_environment_identity(environment):
    return environment_identity(environment) | {
        'toolchain': {key: value for key, value in environment.get('toolchain', {}).items() if key != 'pi_help'},
        'backend_budget': environment.get('backend_budget', {}),
        'device_skus': environment.get('device', {}).get('reported_skus', [])}


def collect_interpreter(executable):
    executable = Path(executable)
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError('configured environment interpreter is unavailable')
    code = ("import runpy,json; m=runpy.run_path(" + repr(str(Path(__file__).resolve())) +
            "); print(json.dumps(m['_collect_current'](),sort_keys=True))")
    result = _run([str(executable), '-c', code], timeout=30)
    if result.get('returncode') != 0:
        raise ValueError('configured environment collector failed')
    try:
        return json.loads(result['stdout'].splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError('configured environment collector returned invalid JSON') from exc


def collect(config: BuilderConfig) -> dict:
    """Run the read-only collector in the operator-selected interpreter."""
    relative = config.raw.get('environment', {}).get('python_relative', '.venv-official/bin/python')
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('configured interpreter must be repository-relative')
    return collect_interpreter(config.repo_root / relative)


def inspect_public_api(executable, symbol):
    """Read a requested public language symbol in the selected interpreter.

    An unlisted symbol is outside the initial view, not proven absent. This
    extends task context without changing the package/toolchain environment.
    """
    if not re.fullmatch(r'[A-Za-z]\w*(?:\.[A-Za-z]\w*)?', symbol) or '__' in symbol:
        raise ValueError('API inspection requires an exact public language symbol')
    code = ('import inspect,json,pypto.language as pl\n'
            'value=pl\n'
            'try:\n'
            ' for name in ' + repr(symbol.split('.')) + ': value=getattr(value,name)\n'
            ' try: signature=str(inspect.signature(value))\n'
            ' except (TypeError,ValueError): signature=None\n'
            ' result={"present":True,"signature":signature}\n'
            'except AttributeError: result={"present":False,"signature":None}\n'
            'print(json.dumps(result,sort_keys=True))\n')
    result = _run([str(executable), '-I', '-c', code], timeout=20)
    if result.get('returncode') != 0:
        return None
    try:
        return json.loads(result['stdout'].splitlines()[-1])
    except (ValueError, IndexError):
        return None
