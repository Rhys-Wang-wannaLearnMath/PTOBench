"""Official Fixture adapter for deepseek_v4.moe_router_flash."""

from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], identity + "_fixture_config"
    )
    # The byte-exact source imports FLASH, regardless of the bound preset.
    alias = ModuleType("config")
    alias.FLASH = config.CASE_CONFIG
    alias.DECODE_BATCH = config.DECODE_BATCH
    alias.DECODE_SEQ = config.DECODE_SEQ
    alias.FP32_NEG_INF = config.FP32_NEG_INF
    alias.EP_WORLD_SIZE = config.EP_WORLD_SIZE
    return common.load_official_fixture_specs(
        sealed / "official/original.py",
        identity + "_official",
        seed,
        config_alias=alias,
        builder_kwargs={"layer_id": config.LAYER_ID},
    )
