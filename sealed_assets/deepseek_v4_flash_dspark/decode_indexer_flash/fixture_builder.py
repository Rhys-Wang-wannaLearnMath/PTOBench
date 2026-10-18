"""Official Fixture adapter for deepseek_v4_flash_dspark.decode_indexer_flash."""

from pathlib import Path
import types

import common


def build(case_dir, sealed, case, seed):
    case_dir = Path(case_dir)
    sealed = Path(sealed)
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], f"fixture_{identity}_public_config",
    )
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config identity disagrees with the Case")
    policy = common.read_json(sealed / "fixture_policy.json")
    expected_seed = policy["fixture_seeds"]["fixture_000"]
    if seed != expected_seed:
        raise ValueError(f"fixture_000 requires seed {expected_seed}, got {seed}")
    original_config = common.load_module(
        sealed / "official" / "original_config.py",
        f"fixture_{identity}_original_config",
    )
    config_alias = types.ModuleType(f"fixture_{identity}_config_alias")
    config_alias.__dict__.update(vars(config))
    config_alias.FLASH = config.CASE_CONFIG
    config_alias.C128_COMPRESSOR_BLOCK_SIZE = original_config.C128_COMPRESSOR_BLOCK_SIZE
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        f"fixture_{identity}_official",
        seed,
        config_alias=config_alias,
    )
