"""Official Fixture adapter for deepseek_v4.prefill_qkv_proj_rope_v2_decode_flash."""

from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    """Bind the selected public preset without changing official input logic."""
    identity = case["case_id"].replace(".", "_")
    cfg = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    # The byte-exact source imports FLASH even for a different bound preset.
    alias = ModuleType("config")
    alias.FLASH = cfg.CASE_CONFIG
    for name in (
        "DECODE_BATCH", "DECODE_SEQ", "PREFILL_BATCH", "PREFILL_SEQ",
        "INT8_SCALE_MAX", "INT8_AMAX_EPS",
    ):
        setattr(alias, name, getattr(cfg, name))
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        identity + "_official",
        seed,
        config_alias=alias,
        builder_args=(cfg.DECODE_BATCH, cfg.DECODE_SEQ),
    )
