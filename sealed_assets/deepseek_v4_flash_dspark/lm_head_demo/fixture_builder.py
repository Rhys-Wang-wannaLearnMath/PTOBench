"""Official Fixture adapter for deepseek_v4_flash_dspark.lm_head_demo."""

from pathlib import Path
from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    case_dir = Path(case_dir)
    sealed = Path(sealed)
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], f"fixture_{identity}_config"
    )
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config Case identity mismatch")
    policy = common.read_json(sealed / "fixture_policy.json")
    if seed != policy["fixture_seeds"]["fixture_000"]:
        raise ValueError("fixture_000 seed mismatch")
    if config.FIXTURE_NUM_TOKENS != config.TEST_TOKENS:
        raise ValueError("official builder default must match the fixed Fixture binding")
    alias = ModuleType(f"fixture_{identity}_config_alias")
    alias.FLASH = config.CASE_CONFIG
    alias.DECODE_TOKENS = config.DECODE_TOKENS
    specs = common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        f"fixture_{identity}_official",
        seed,
        config_alias=alias,
    )
    if [spec.name for spec in specs] != list(config.TENSOR_SHAPES):
        raise ValueError("official Fixture ABI order mismatch")
    for spec in specs:
        if list(spec.shape) != config.TENSOR_SHAPES[spec.name]:
            raise ValueError(f"official Fixture shape mismatch: {spec.name}")
        if str(spec.dtype) != f"torch.{config.TENSOR_DTYPES[spec.name]}":
            raise ValueError(f"official Fixture dtype mismatch: {spec.name}")
    return specs
