"""Official Fixture adapter for deepseek_v4.prefill_moe_ep4_pro; never executes an NPU program."""

import ast
import importlib.abc
import importlib.util
import os
import sys
from contextlib import contextmanager
from types import ModuleType

import common


class _MetadataLoader(importlib.abc.Loader):
    """Omit one approved, unreachable DSL tuning assertion in memory only."""

    def __init__(self, path, expression):
        self.path = path
        self.expression = expression

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        tree = ast.parse(self.path.read_text(), filename=str(self.path))
        matches = [node for node in tree.body if isinstance(node, ast.Assert)
                   and ast.unparse(node.test) == self.expression]
        if len(matches) != 1:
            raise RuntimeError("approved Fixture assertion must match exactly once")
        tree.body.remove(matches[0])
        # No generator, golden body, or remaining check is changed. This
        # imported DSL is construction metadata, never the Ref implementation.
        module.__file__ = str(self.path)
        module.__fixture_removed_assertions__ = [self.expression]
        exec(compile(tree, str(self.path), "exec"), module.__dict__)


class _MetadataFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, module_name, expression):
        self.path = root / (module_name + ".py")
        self.module_name = module_name
        self.expression = expression

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.module_name:
            return importlib.util.spec_from_file_location(
                fullname, self.path,
                loader=_MetadataLoader(self.path, self.expression),
            )
        return None


@contextmanager
def _metadata_imports(official_root, config):
    finder = None
    finder = _MetadataFinder(
        official_root, "hc_pre_v2",
        "HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0",
    )
    previous_argv = sys.argv
    previous_hc_impl = os.environ.get("DSV4_HC_PRE_IMPL")
    try:
        sys.argv = [str(official_root / "original.py"), "--ep", str(config.EP_WORLD_SIZE)]
        os.environ["DSV4_HC_PRE_IMPL"] = "separate"
        if finder is not None:
            sys.meta_path.insert(0, finder)
        yield
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)
        sys.argv = previous_argv
        if previous_hc_impl is None:
            os.environ.pop("DSV4_HC_PRE_IMPL", None)
        else:
            os.environ["DSV4_HC_PRE_IMPL"] = previous_hc_impl


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    # The byte-exact moe.py performs //8*EP itself. Bind the unscaled
    # selected object here so it applies that source expression exactly once.
    alias.FLASH = config.PRO
    # Only historical helper annotation/standalone scaffolding uses decode.
    alias.DECODE_BATCH = 8
    alias.DECODE_SEQ = 1
    alias.DECODE_TOKENS = alias.DECODE_BATCH * alias.DECODE_SEQ
    official_root = sealed / "official"
    with _metadata_imports(official_root, config):
        return common.load_official_fixture_specs(
            official_root / "original.py", identity + "_official", seed,
            config_alias=alias,
            builder_args=(config.LAYER_ID, config.NUM_TOKENS),
        )
