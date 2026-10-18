"""Official Fixture adapter for deepseek_v4.prefill_swa_sparse_attn_pro."""

from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "prefill_swa_sparse_attn_pro_fixture_config"
    )
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    specs = common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "prefill_swa_sparse_attn_pro_official",
        seed,
        config_alias=alias,
    )
    start_pos = next(spec for spec in specs if spec.name == "start_pos")
    if start_pos.value != config.START_POS:
        raise ValueError("Official start_pos differs from the public fixed contract")
    return specs
