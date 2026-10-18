"""Official TensorSpec fixture builder for the DEMO gate Case."""

import sys

import torch
from golden import ScalarSpec

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "deepseek_v4_gate_demo_fixture_config"
    )
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    specs = common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "deepseek_v4_gate_demo_official",
        seed,
        config_alias=alias,
    )
    fixed_scalars = {
        "layer_id": config.LAYER_ID,
        "num_tokens": config.NUM_TOKENS,
    }
    return [
        ScalarSpec(spec.name, torch.int32, fixed_scalars[spec.name])
        if spec.name in fixed_scalars
        else spec
        for spec in specs
    ]
