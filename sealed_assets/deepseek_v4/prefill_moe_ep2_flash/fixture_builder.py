"""Official Fixture adapter for deepseek_v4.prefill_moe_ep2_flash."""

import sys
from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"],
        "fixture_config_deepseek_v4_prefill_moe_ep2_flash",
    )
    original_config = common.load_module(
        sealed / "official" / "original_config.py",
        "official_config_deepseek_v4_prefill_moe_ep2_flash",
    )
    alias = ModuleType("config")
    alias.__file__ = original_config.__file__
    alias.__dict__.update(
        {name: value for name, value in vars(original_config).items()
         if not name.startswith("_")}
    )
    alias.__dict__.update(
        {name: value for name, value in vars(config).items()
         if not name.startswith("_")}
    )
    alias.FLASH = config.CASE_CONFIG
    previous_argv = sys.argv
    sys.argv = [case["case_id"], "--ep", str(config.EP_WORLD_SIZE)]
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            "official_deepseek_v4_prefill_moe_ep2_flash",
            seed,
            config_alias=alias,
        )
    finally:
        sys.argv = previous_argv
