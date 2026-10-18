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
