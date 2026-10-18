"""Official Fixture adapter for deepseek_v4_flash_mtp.prefill_cp_hca_cp4_flash (FLASH CP4 HCA)."""

import ast
import hashlib
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


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    assert case["case_id"] == "deepseek_v4_flash_mtp.prefill_cp_hca_cp4_flash"
    assert config.CASE_CONFIG.name == "flash"
    root = sealed / "official"
    historical = common.load_module(root / "original_config.py", identity + "_original_config")
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    # These names serve only unused historical DSL imports/defaults. Every live
    # HCA Fixture dimension, capacity, quantization value and preset is public.
    for name in (
        "DECODE_BATCH", "DECODE_SEQ", "CSA_STATE_PHYSICAL_BLOCKS",
        "CSA_INNER_STATE_PHYSICAL_BLOCKS", "PREFILL_IDX_BLOCK_NUM",
        "PREFILL_IDX_MAX_BLOCKS", "PREFILL_ORI_BLOCK_NUM",
    ):
        setattr(alias, name, getattr(historical, name))
    import importlib.abc
    import importlib.util
    class _QKVFixtureLoader(importlib.abc.Loader):

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            path = root / 'qkv_proj_rope.py'
            source = path.read_bytes()
            assert hashlib.sha256(source).hexdigest() == 'fe73c2d238642ab9f8561646c8a4e2e8234c8e20ce6da218e8ce034609a00150'
            exec(compile(source, str(path), 'exec'), module.__dict__)
            tree = ast.parse(path.read_text())
            function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_tensor_specs'))
            compatibility = _FixtureUniformCompatibility()
            function = compatibility.visit(function)
            assert compatibility.count == 8
            unit = ast.Module(body=[function], type_ignores=[])
            ast.fix_missing_locations(unit)
            namespace = dict(vars(module))
            namespace['_historical_bf16_uniform'] = _historical_bf16_uniform
            exec(compile(unit, str(path), 'exec'), namespace)
            module.build_tensor_specs = namespace['build_tensor_specs']
    class _QKVFixtureFinder(importlib.abc.MetaPathFinder):

        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'qkv_proj_rope':
                return importlib.util.spec_from_file_location(fullname, root / 'qkv_proj_rope.py', loader=_QKVFixtureLoader())
            return None
    finder = _QKVFixtureFinder()
    previous_argv = sys.argv
    try:
        sys.argv = [str(root / "original.py"), "--cp", str(config.CP_SIZE)]
        sys.meta_path.insert(0, finder)
        with torch.random.fork_rng(devices=[]):
            return common.load_official_fixture_specs(
                root / "original.py", identity + "_official", seed,
                config_alias=alias, builder_args=(config.CP_SIZE,),
            )
    finally:
        sys.meta_path.remove(finder)
        sys.argv = previous_argv
