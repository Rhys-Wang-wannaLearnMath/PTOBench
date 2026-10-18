"""Thin FLASH adapter for deepseek_v4_flash_dspark.prefill_swa_flash's byte-exact official specs."""
from pathlib import Path
import types
import common


def build(case_dir, sealed, case, seed):
    case_dir, sealed = Path(case_dir), Path(sealed)
    case_id = case["case_id"]
    identity = case_id.replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], f"fixture_{identity}_public_config"
    )
    if case_id != config.CASE_ID:
        raise ValueError(f"public config CASE_ID={config.CASE_ID} disagrees with {case_id}")
    policy = common.read_json(sealed / "fixture_policy.json")
    fixed_seed = policy["fixture_seeds"]["fixture_000"]
    if seed != fixed_seed or seed != config.FIXTURE_SEED:
        raise ValueError(f"fixture_000 requires public/policy seed {config.FIXTURE_SEED}, got {seed}")
    alias = types.ModuleType(f"fixture_{identity}_config_alias")
    alias.__dict__.update(vars(config))
    # The preserved entry and helpers import this historical name.
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official/original.py", f"fixture_{identity}_official", seed,
        config_alias=alias,
        builder_kwargs={"start_pos": config.START_POS, "num_tokens": config.T},
    )
