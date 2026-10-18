"""Official TensorSpec fixture builder for deepseek_v4.moe_dispatch_demo."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "moe_dispatch_demo_fixture_config"
    )
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "moe_dispatch_demo_official",
        seed,
        config_alias=alias,
    )
