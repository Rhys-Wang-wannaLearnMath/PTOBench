"""Adapter for the byte-exact DeepSeek-V4 dispatch Fixture builder."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "dispatch_pro_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "dispatch_pro_official",
        seed,
        config_alias=alias,
    )
