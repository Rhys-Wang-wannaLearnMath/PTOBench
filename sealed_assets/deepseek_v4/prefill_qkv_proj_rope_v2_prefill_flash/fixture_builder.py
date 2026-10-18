"""Official Fixture adapter for deepseek_v4.prefill_qkv_proj_rope_v2_prefill_flash."""

from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    cfg = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    # Only the historical import name is adapted; selected source stays byte-exact.
    alias = ModuleType("config")
    alias.FLASH = cfg.CASE_CONFIG
    for name in ("DECODE_BATCH", "DECODE_SEQ", "PREFILL_BATCH", "PREFILL_SEQ",
                 "INT8_SCALE_MAX", "INT8_AMAX_EPS"):
        setattr(alias, name, getattr(cfg, name))
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        identity + "_official",
        seed,
        config_alias=alias,
        builder_args=(cfg.PREFILL_BATCH, cfg.PREFILL_SEQ),
    )
