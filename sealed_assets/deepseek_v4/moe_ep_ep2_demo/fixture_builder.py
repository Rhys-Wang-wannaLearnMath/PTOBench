"""Thin adapter for the byte-exact official TensorSpec builder."""
from contextlib import contextmanager
import importlib.abc
import importlib.util
from types import ModuleType
import sys

import common


class _GateLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        import ast
        tree = ast.parse(self.path.read_text(), filename=str(self.path))
        removed = [n for n in tree.body if isinstance(n, ast.Assert)
                   and ast.unparse(n.test) == "N_EXPERTS % GATE_N_TILE == 0"]
        if len(removed) != 1:
            raise RuntimeError("official DEMO gate assertion was not found")
        tree.body = [n for n in tree.body if n not in removed]
        module.__file__ = str(self.path)
        module.__manual_removed_assertions__ = ["N_EXPERTS % GATE_N_TILE == 0"]
        exec(compile(tree, str(self.path), "exec"), module.__dict__)


class _GateFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.path = path
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "gate":
            return importlib.util.spec_from_file_location(fullname, self.path, loader=_GateLoader(self.path))
        return None


@contextmanager
def _gate_compatibility(official_root, enabled):
    finder = _GateFinder(official_root / "gate.py") if enabled else None
    if finder is not None:
        sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    preset = case["case_id"].rsplit("_", 1)[-1].upper()
    selected = getattr(config, preset)
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    # The historical entry imports config.FLASH and applies the EP rewrite once.
    alias.FLASH = selected
    official_root = sealed / "official"
    previous_argv = sys.argv
    try:
        sys.argv = [str(official_root / "original.py"), "--ep", str(config.EP_WORLD_SIZE)]
        with _gate_compatibility(official_root, config.N_EXPERTS_GLOBAL < 16):
            return common.load_official_fixture_specs(
                official_root / "original.py", identity + "_official", seed,
                config_alias=alias, builder_kwargs={"layer_id": config.LAYER_ID})
    finally:
        sys.argv = previous_argv
