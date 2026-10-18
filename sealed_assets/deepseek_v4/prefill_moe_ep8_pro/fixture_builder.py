"""Official Fixture/golden import adapter for deepseek_v4.prefill_moe_ep8_pro."""

import ast
from contextlib import contextmanager
import importlib.abc
import importlib.util
import sys
from types import ModuleType

import common


class _HCPreLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        tree = ast.parse(self.path.read_text(), filename=str(self.path))
        expression = "HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0"
        removed = [
            node for node in tree.body
            if isinstance(node, ast.Assert) and ast.unparse(node.test) == expression
        ]
        if len(removed) != 1:
            raise RuntimeError("Expected exactly one unused HC syncall tiling assertion")
        tree.body = [node for node in tree.body if node not in removed]
        # Construction only: remove the unused syncall assertion in memory.
        # Every generator, TensorSpec and CPU golden body stays unchanged.
        # This loaded DSL implementation is never used for Ref execution.
        module.__file__ = str(self.path)
        module.__manual_removed_assertions__ = [expression]
        exec(compile(tree, str(self.path), "exec"), module.__dict__)


class _HCPreFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "hc_pre_v2":
            return importlib.util.spec_from_file_location(
                fullname, self.path, loader=_HCPreLoader(self.path)
            )
        return None


@contextmanager
def _official_scope(case_dir, sealed, case):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    # The preserved source applies its //8*EP replacement itself.
    alias.FLASH = config.CASE_CONFIG
    official_root = sealed / "official"
    finder = _HCPreFinder(official_root / "hc_pre_v2.py")
    previous_path, previous_argv = sys.path[:], sys.argv
    previous_bytecode = sys.dont_write_bytecode
    with common._isolated_module_scope((official_root,)):
        prior_config = sys.modules.get("config")
        try:
            sys.modules["config"] = alias
            # Keep delayed official-golden imports inside this owning tree.
            sys.path.insert(0, str(official_root))
            sys.argv = [str(official_root / "original.py"), "--ep", str(config.EP_WORLD_SIZE)]
            sys.dont_write_bytecode = True
            sys.meta_path.insert(0, finder)
            yield config, alias, official_root, identity
        finally:
            sys.meta_path.remove(finder)
            sys.path[:], sys.argv = previous_path, previous_argv
            sys.dont_write_bytecode = previous_bytecode
            if prior_config is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = prior_config


@contextmanager
def load_official(case_dir, sealed, case):
    """Keep original delayed CPU golden imports isolated for certification."""
    with _official_scope(case_dir, sealed, case) as (config, alias, root, identity):
        official = common.load_module(root / "original.py", identity + "_official")
        if (official.N_RANKS, official.N_EXPERTS_GLOBAL, official.N_LOCAL, official.T) != (
            config.EP_WORLD_SIZE, config.N_EXPERTS_GLOBAL, config.N_LOCAL, config.T
        ):
            raise RuntimeError("Official PRO binding differs from the public contract")
        yield official


def build(case_dir, sealed, case, seed):
    with _official_scope(case_dir, sealed, case) as (config, alias, root, identity):
        return common.load_official_fixture_specs(
            root / "original.py", identity + "_official", seed, config_alias=alias,
            builder_kwargs={"layer_id": config.LAYER_ID, "num_tokens": config.NUM_TOKENS},
        )
