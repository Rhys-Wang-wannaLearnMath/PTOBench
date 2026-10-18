"""Official Fixture adapter for deepseek_v4_flash_dspark.dspark_markov_demo."""

from pathlib import Path
from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    case_dir = Path(case_dir)
    sealed = Path(sealed)
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], f"fixture_{identity}_public_config",
    )
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config and Case identity disagree")
    policy = common.read_json(sealed / "fixture_policy.json")
    expected_seed = policy["fixture_seeds"]["fixture_000"]
    if seed != expected_seed:
        raise ValueError(f"fixture_000 requires seed {expected_seed}, got {seed}")
    original_config = common.load_module(
        sealed / "official" / "original_config.py",
        f"fixture_{identity}_original_config",
    )
    alias = ModuleType(f"fixture_{identity}_config_alias")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    # Import-time defaults for the preserved helper's standalone CLI only.
    alias.DECODE_BATCH = original_config.DECODE_BATCH
    alias.TP = original_config.TP
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        f"fixture_{identity}_official",
        seed,
        config_alias=alias,
        builder_args=(config.FIXTURE_BATCH,),
    )
