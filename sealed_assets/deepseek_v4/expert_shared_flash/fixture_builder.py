"""Official TensorSpec fixture adapter for the FLASH shared expert Case."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "expert_shared_flash_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py", "expert_shared_flash_official", seed,
        config_alias=alias,
    )
