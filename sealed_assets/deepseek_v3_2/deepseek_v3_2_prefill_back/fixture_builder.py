"""Adapter around the frozen official TensorSpec builder."""
import sys
import common

def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "prefill_back_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py", "prefill_back_official", seed,
        config_alias=alias,
    )
