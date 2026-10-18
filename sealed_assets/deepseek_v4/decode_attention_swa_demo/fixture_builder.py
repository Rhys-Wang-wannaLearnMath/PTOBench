"""Official TensorSpec adapter for the DEMO SWA contract."""

import sys

import common
import torch


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "decode_attention_swa_demo_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    specs = common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "decode_attention_swa_demo_official",
        seed,
        config_alias=alias,
    )
    # Bind the approved heterogeneous (127, 126, ...) start-position fixture.
    start = torch.tensor([config.START_POS, max(config.START_POS - 1, 0)], dtype=torch.int32)
    start = start.repeat((config.B + start.numel() - 1) // start.numel())[:config.B].clone()
    seq = start.to(torch.int64) + config.S
    used = torch.minimum(seq, torch.full_like(seq, config.WIN)).to(torch.int32)
    for spec in specs:
        if spec.name == "start_pos":
            spec.init_value = start
        elif spec.name == "seqused_kv":
            spec.init_value = used
    return specs
