"""Official TensorSpec adapter for the PRO ratio-128 HCA contract."""

import sys

import common
import torch


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "decode_attention_hca_pro_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    specs = common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "decode_attention_hca_pro_official",
        seed,
        config_alias=alias,
    )
    start = torch.tensor([config.START_POS, max(config.START_POS - 1, 0)], dtype=torch.int32)
    start = start.repeat((config.B + start.numel() - 1) // start.numel())[:config.B].clone()
    seq = start.to(torch.int64) + config.S
    used = torch.where(seq <= config.WIN, seq, config.WIN + seq // config.COMPRESS_RATIO).to(torch.int32)
    for spec in specs:
        if spec.name == "start_pos":
            spec.init_value = start
        elif spec.name == "seqused_kv":
            spec.init_value = used
    return specs
