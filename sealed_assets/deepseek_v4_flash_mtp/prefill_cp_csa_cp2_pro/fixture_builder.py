"""Case-local adapter for the byte-exact official TensorSpec builder."""
from types import ModuleType
import ast
import importlib.abc
import importlib.util
import sys

import common


def _config_alias(case_dir, sealed, case, identity):
    public = common.load_module(case_dir / case["config"], identity + "_public_config")
    original = common.load_module(sealed / "official" / "original_config.py", identity + "_original_config")
    alias = ModuleType("config")
    for source in (original, public):
        for key, value in vars(source).items():
            if not key.startswith("__"):
                setattr(alias, key, value)
    # The preserved original.py imports the historical default name FLASH.
    alias.FLASH = public.CASE_CONFIG
    return public, alias


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    public, alias = _config_alias(case_dir, sealed, case, identity)
    official = sealed / "official" / "original.py"
    return common.load_official_fixture_specs(
        official,
        identity + "_official",
        seed,
        config_alias=alias,
        builder_kwargs={"cp_size": public.CP_SIZE},
    )


class _HCPreLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        tree = ast.parse(self.path.read_text(), filename=str(self.path))
        expression = "HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0"
        removed = [node for node in tree.body if isinstance(node, ast.Assert) and ast.unparse(node.test) == expression]
        if len(removed) != 1:
            raise AssertionError(f"expected one PRO HC split assertion, found {len(removed)}")
        tree.body = [node for node in tree.body if node not in removed]
        module.__file__ = str(self.path)
        exec(compile(tree, str(self.path), "exec"), module.__dict__)


class _HCPreFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "hc_pre":
            return importlib.util.spec_from_file_location(fullname, self.path, loader=_HCPreLoader(self.path))
        return None


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    public, alias = _config_alias(case_dir, sealed, case, identity)
    finder = _HCPreFinder(sealed / "official" / "hc_pre.py")
    sys.meta_path.insert(0, finder)
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            identity + "_official",
            seed,
            config_alias=alias,
            builder_kwargs={"cp_size": public.CP_SIZE},
        )
    finally:
        sys.meta_path.remove(finder)
