"""Fixture adapter for deepseek_v4_flash_dspark.moe_flash."""
from pathlib import Path
from types import ModuleType
import sys

import common


def build(case_dir, sealed, case, seed):
    case_dir = Path(case_dir)
    sealed = Path(sealed)
    policy = common.read_json(sealed / "fixture_policy.json")
    if seed != policy["fixture_seeds"]["fixture_000"]:
        raise ValueError("fixture_000 seed mismatch")
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], f"fixture_{identity}_config")
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config Case identity mismatch")
    if (config.LAYER_ID, config.NUM_TOKENS, config.BALANCED_ROUTING) != (
        0, config.MOE_TOKENS, False
    ):
        raise ValueError("public scalar bindings disagree with the official builder defaults")
    if config.RECV_MAX != config.EP * config.MOE_TOKENS:
        raise ValueError("public receive capacity disagrees with the official EP binding")
    if config.N_EXPERTS_GLOBAL != config.CASE_CONFIG.n_routed_experts // 16 * config.EP:
        raise ValueError("public expert count disagrees with the official normalization")
    original_config = common.load_module(
        sealed / "official" / "original_config.py", f"fixture_{identity}_original_config"
    )
    alias = ModuleType(f"fixture_{identity}_config_alias")
    alias.__dict__.update(vars(config))
    # original.py normalizes this 256-expert model object once to the EP2 ABI.
    # MOE_TOKENS already reflects the source's earlier EP16 config evaluation.
    alias.FLASH = config.CASE_CONFIG
    for name in (
        "PREFILL_BATCH", "PREFILL_SEQ", "C4A_COMPRESSOR_BLOCK_SIZE",
        "C128_COMPRESSOR_BLOCK_SIZE", "DECODE_START_POS",
    ):
        setattr(alias, name, getattr(original_config, name))
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], "--ep", str(config.EP)]
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            f"fixture_{identity}_official",
            seed,
            config_alias=alias,
        )
    finally:
        sys.argv = previous_argv
