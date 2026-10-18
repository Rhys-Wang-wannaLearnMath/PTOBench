"""Build official fixture specs with this Case's fixed preset binding."""

import types

import common


def build(case_dir, sealed, case, seed):
    public_config = common.load_module(
        case_dir / case["config"],
        "public_config_deepseek_v4_prefill_qkv_proj_rope_demo",
    )
    config_alias = types.ModuleType("config")
    config_alias.FLASH = public_config.CASE_CONFIG
    config_alias.INT8_AMAX_EPS = public_config.INT8_AMAX_EPS
    config_alias.INT8_SCALE_MAX = public_config.INT8_SCALE_MAX
    config_alias.PREFILL_BATCH = public_config.PREFILL_BATCH
    config_alias.PREFILL_SEQ = public_config.PREFILL_SEQ
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "official_deepseek_v4_prefill_qkv_proj_rope_demo",
        seed, config_alias=config_alias,
    )
