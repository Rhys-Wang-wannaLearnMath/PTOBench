"""Official Fixture adapter for deepseek_v4.moe_expert_flash (FLASH)."""

from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "config_moe_expert_flash_fixture",
    )
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py", "official_moe_expert_flash", seed,
        config_alias=alias,
    )
