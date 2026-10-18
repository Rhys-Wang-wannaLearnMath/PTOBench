"""Official TensorSpec fixture builder for the PRO compatibility Case."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "qkv_proj_rope_compat_pro_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    previous_dependency = sys.modules.pop("decode_qkv_proj_rope", None)
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            "qkv_proj_rope_compat_pro_official",
            seed,
            config_alias=alias,
        )
    finally:
        sys.modules.pop("decode_qkv_proj_rope", None)
        if previous_dependency is not None:
            sys.modules["decode_qkv_proj_rope"] = previous_dependency
