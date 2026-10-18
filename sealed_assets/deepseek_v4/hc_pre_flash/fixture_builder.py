"""Official TensorSpec adapter for the FLASH hyper-connections Case."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "hc_pre_flash_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    alias.DECODE_BATCH = config.DECODE_BATCH
    alias.DECODE_SEQ = config.DECODE_SEQ
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "hc_pre_flash_official",
        seed,
        config_alias=alias,
    )
