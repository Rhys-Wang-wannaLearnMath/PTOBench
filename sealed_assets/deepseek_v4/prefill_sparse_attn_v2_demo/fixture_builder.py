"""Case-local adapter for the byte-exact v2 official Fixture builder."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "prefill_sparse_attn_v2_demo_public_config")
    if config.DEFAULT_COMPRESS_RATIO != 4:
        raise ValueError("deepseek_v4.prefill_sparse_attn_v2_demo requires compress_ratio=4")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    previous_rope = sys.modules.pop("rope_tables", None)
    official_root = str(sealed / "official")
    sys.path.insert(0, official_root)
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            "prefill_sparse_attn_v2_demo_official",
            seed,
            config_alias=alias,
        )
    finally:
        sys.path.remove(official_root)
        sys.modules.pop("rope_tables", None)
        if previous_rope is not None:
            sys.modules["rope_tables"] = previous_rope
