"""Official TensorSpec adapter for deepseek_v4.hc_head_demo."""

import sys

import common


def build(case_dir, sealed, case, seed):
    case_id = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], f"{case_id}_public_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py", f"{case_id}_official", seed,
        config_alias=alias,
    )
