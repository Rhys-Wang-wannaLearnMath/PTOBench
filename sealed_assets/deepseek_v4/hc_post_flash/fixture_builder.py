"""Official Fixture adapter for deepseek_v4.hc_post_flash."""

from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "deepseek_v4_hc_post_flash_fixture_config",
    )
    original_config = common.load_module(
        sealed / "official" / "original_config.py",
        "deepseek_v4_hc_post_flash_original_config",
    )
    alias = ModuleType("config")
    alias.FLASH = config.CASE_CONFIG
    alias.DECODE_BATCH = config.DECODE_BATCH
    alias.DECODE_SEQ = config.DECODE_SEQ
    # Import/assert compatibility only; the selected builder receives decode B, S.
    alias.PREFILL_BATCH = original_config.PREFILL_BATCH
    alias.PREFILL_SEQ = original_config.PREFILL_SEQ
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "deepseek_v4_hc_post_flash_official",
        seed,
        config_alias=alias,
        builder_args=(config.B, config.S),
    )
