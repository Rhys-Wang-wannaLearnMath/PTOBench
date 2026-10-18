"""Official TensorSpec fixture adapter for the DEMO prefill HC-post Case."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "prefill_hc_post_demo_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    alias.PREFILL_BATCH = config.PREFILL_BATCH
    alias.PREFILL_SEQ = config.PREFILL_SEQ
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "prefill_hc_post_demo_official",
        seed,
        config_alias=alias,
    )
