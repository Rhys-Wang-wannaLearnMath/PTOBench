"""Official Fixture adapter for deepseek_v4_flash_mtp.sample_pro."""

from pathlib import Path
from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    """Construct official specs without materializing a Fixture."""
    case_dir = Path(case_dir)
    sealed = Path(sealed)
    if case["case_id"] != "deepseek_v4_flash_mtp.sample_pro":
        raise ValueError("Fixture adapter and Case identity disagree")
    identity = case["case_id"].replace(".", "_")
    config = common.load_isolated_module(
        case_dir / case["config"], f"fixture_{identity}_public_config",
    )
    if config.FIXTURE_TEMPERATURE is not None or config.FIXTURE_TOP_K is not None:
        raise ValueError("Official sampling Fixture requires mixed-vector defaults")
    policy = common.read_json(sealed / "fixture_policy.json")
    if seed != policy["fixture_seeds"]["fixture_000"]:
        raise ValueError("Seed disagrees with the canonical fixture_000 policy")
    alias = ModuleType(f"fixture_{identity}_config_alias")
    alias.DECODE_TOKENS = config.DECODE_TOKENS
    alias.FLASH = config.CASE_CONFIG
    alias.FP32_NEG_INF = config.FP32_NEG_INF
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py", f"official_{identity}",
        seed, config_alias=alias,
    )
