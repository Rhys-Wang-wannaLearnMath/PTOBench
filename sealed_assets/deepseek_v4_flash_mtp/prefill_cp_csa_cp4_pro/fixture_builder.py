"""Official Fixture adapter for deepseek_v4_flash_mtp.prefill_cp_csa_cp4_pro.

Load the byte-exact selected source with its selected public contract.
Retain only the approved unused HC-pre import assertion guard.
"""

import ast
import importlib.abc
import importlib.util
import sys
from types import ModuleType

import torch

import common


class _HCPreLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        tree = ast.parse(self.path.read_text(), filename=str(self.path))
        expression = 'HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0'
        removed = [n for n in tree.body if isinstance(n, ast.Assert) and ast.unparse(n.test) == expression]
        assert len(removed) == 1
        tree.body = [n for n in tree.body if n not in removed]
        module.__file__ = str(self.path)
        module.__manual_removed_assertions__ = [expression]
        # Only import unused DSL specialization; never execute its runtime.
        # All generators and golden bodies remain byte-exact.
        exec(compile(tree, str(self.path), 'exec'), module.__dict__)


class _HCPreFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'hc_pre':
            return importlib.util.spec_from_file_location(fullname, self.path, loader=_HCPreLoader(self.path))
        return None


def build(case_dir, sealed, case, seed):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_fixture_config')
    root = sealed / 'official'
    historical = common.load_module(root / 'original_config.py',
                                    identity + '_original_config')
    # Historical fields serve only unused official annotations. Public bindings win.
    alias = ModuleType('config')
    alias.__dict__.update(vars(historical))
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    finder = _HCPreFinder(root / 'hc_pre.py')
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(root / 'original.py'), '--cp', str(config.CP_SIZE)]
        if finder is not None:
            sys.meta_path.insert(0, finder)
        with torch.random.fork_rng(devices=[]):
            return common.load_official_fixture_specs(
                root / 'original.py', identity + '_official', seed,
                config_alias=alias, builder_kwargs={'cp_size': config.CP_SIZE})
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)
        sys.argv[:] = previous_argv
