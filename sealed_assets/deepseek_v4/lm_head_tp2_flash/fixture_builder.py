"""Official TensorSpec fixture builder for the lm_head_tp2_flash LM-head Case."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "lm_head_tp2_flash_fixture_config"
    )
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "lm_head_tp2_flash_official",
        seed,
        config_alias=alias,
    )
