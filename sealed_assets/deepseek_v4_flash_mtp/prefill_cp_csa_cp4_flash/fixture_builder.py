"""Official Fixture adapter for deepseek_v4_flash_mtp.prefill_cp_csa_cp4_flash.

Load the byte-exact selected source with its selected public contract.
Retain only the eight approved historical QKV BF16 uniform initializers.
"""

import ast
import importlib.abc
import importlib.util
import sys
from types import ModuleType

import torch

import common


def _historical_bf16_uniform(shape, lower, upper):
    value = torch.empty(shape, dtype=torch.float32).uniform_(lower, upper).to(torch.bfloat16)
    return torch.where(value >= upper, torch.full_like(value, lower), value)


class _FixtureUniformCompatibility(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'uniform_'
                and isinstance(node.func.value, ast.Call)):
            allocation = node.func.value
            if ast.unparse(allocation.func) == 'torch.empty':
                dtype = next((kw.value for kw in allocation.keywords if kw.arg == 'dtype'), None)
                if dtype is not None and ast.unparse(dtype) == 'torch.bfloat16':
                    assert len(allocation.args) == 1 and len(node.args) == 2 and not node.keywords
                    assert len(allocation.keywords) == 1
                    self.count += 1
                    return ast.Call(func=ast.Name(id='_historical_bf16_uniform', ctx=ast.Load()),
                                    args=[allocation.args[0], *node.args], keywords=[])
        return node


class _QKVFixtureLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Preserve every source function except the approved Fixture initializer calls.
        source = self.path.read_text()
        exec(compile(source, str(self.path), 'exec'), module.__dict__)
        tree = ast.parse(source)
        function = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == 'build_tensor_specs')
        compatibility = _FixtureUniformCompatibility()
        function = compatibility.visit(function)
        assert compatibility.count == 8
        unit = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(unit)
        namespace = dict(vars(module))
        namespace['_historical_bf16_uniform'] = _historical_bf16_uniform
        exec(compile(unit, str(self.path), 'exec'), namespace)
        module.build_tensor_specs = namespace['build_tensor_specs']


class _QKVFixtureFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'qkv_proj_rope':
            return importlib.util.spec_from_file_location(
                fullname, self.path, loader=_QKVFixtureLoader(self.path))
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
    finder = _QKVFixtureFinder(root / 'qkv_proj_rope.py')
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
