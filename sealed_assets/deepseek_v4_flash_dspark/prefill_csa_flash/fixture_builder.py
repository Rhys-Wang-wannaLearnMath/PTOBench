"""Official Fixture adapter for deepseek_v4_flash_dspark.prefill_csa_flash."""
from __future__ import annotations

from pathlib import Path
import types

import common


def build(case_dir, sealed, case, seed):
    case_dir, sealed = Path(case_dir), Path(sealed)
    case_id = case["case_id"]
    identity = case_id.replace(".", "_")
    config = common.load_isolated_module(
        case_dir / case["config"], f"fixture_{identity}_public_config",
    )
    if config.CASE_ID != case_id:
        raise ValueError("public config CASE_ID mismatch")
    policy = common.read_json(sealed / "fixture_policy.json")
    if seed != policy["fixture_seeds"]["fixture_000"]:
        raise ValueError("fixture seed mismatch")

    official_root = sealed / "official"
    original_config = common.load_isolated_module(
        official_root / "original_config.py", f"fixture_{identity}_original_config",
    )
    # Byte-exact helpers also import constants for unrelated legacy test paths.
    # Only the sealed alias retains those; all fixed CSA values come from public config.
    alias = types.ModuleType(f"fixture_{identity}_config_alias")
    alias.__dict__.update(vars(original_config))
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    builder_kwargs = {"start_pos": config.START_POS, "num_tokens": config.NUM_TOKENS}
    return common.load_official_fixture_specs(
        official_root / "original.py", f"fixture_{identity}_official", seed,
        config_alias=alias, builder_kwargs=builder_kwargs,
    )
