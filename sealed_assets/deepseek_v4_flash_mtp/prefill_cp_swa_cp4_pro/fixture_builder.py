"""Full official SWA PRO Fixture; ignore only an unused syncall assertion."""
import ast
import importlib.abc
import importlib.util
import sys
from types import ModuleType

import common
import torch


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
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_fixture_config')
    alias = ModuleType('config')
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    root = sealed / 'official'
    finder = _HCPreFinder(root / 'hc_pre.py')
    previous_argv = sys.argv
    try:
        sys.argv = [str(root / 'original.py'), '--cp', str(config.CP_SIZE)]
        if finder is not None:
            sys.meta_path.insert(0, finder)
        with torch.random.fork_rng(devices=[]):
            result = common.load_official_fixture_specs(
                root / 'original.py', identity + '_official', seed, config_alias=alias,
                builder_kwargs={'cp_size': config.CP_SIZE})
            specs, _context = result
            return specs
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)
        sys.argv = previous_argv

