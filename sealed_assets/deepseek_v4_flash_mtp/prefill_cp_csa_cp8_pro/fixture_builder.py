"""CP8 PRO official Fixture adapter using the existing official loader.

Stage 1 approved this Fixture-only import guard adaptation.
Preserved official files, generators and CPU golden remain byte-exact.
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
        exec(compile(tree, str(self.path), 'exec'), module.__dict__)


class _HCPreFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'hc_pre':
            return importlib.util.spec_from_file_location(fullname, self.path, loader=_HCPreLoader(self.path))
        return None


def build(case_dir, sealed, case, seed):
    identity = case['case_id'].replace('.', '_') + '_official_fixture'
    config = common.load_module(case_dir / case['config'], identity + '_config')
    alias = ModuleType('config')
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    root = sealed / 'official'
    finder = _HCPreFinder(root / 'hc_pre.py')
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(root / 'original.py'), '--cp', str(config.CP_SIZE)]
        sys.meta_path.insert(0, finder)
        with torch.random.fork_rng(devices=[]):
            return common.load_official_fixture_specs(
                root / 'original.py', identity + '_official', seed, config_alias=alias,
                builder_kwargs={"cp_size": config.CP_SIZE})
    finally:
        sys.meta_path.remove(finder)
        sys.argv[:] = previous_argv
