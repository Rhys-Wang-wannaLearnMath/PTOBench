"""Official TensorSpec adapter for the PRO compressor Case."""
import sys
import common
import torch

def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "deepseek_v4_decode_compressor_ratio4_pro_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    specs = common.load_official_fixture_specs(sealed / "official" / "original.py", "deepseek_v4_decode_compressor_ratio4_pro_official", seed, config_alias=alias)
    pattern = torch.tensor([0, config.COMPRESS_RATIO - config.DECODE_SEQ, config.COMPRESS_RATIO - 1, config.COMPRESS_RATIO * 2 - 1], dtype=torch.int32)
    start = pattern.repeat((config.B + pattern.numel() - 1) // pattern.numel())[:config.B]
    for spec in specs:
        if spec.name == "start_pos":
            spec.init_value = start.clone()
    return specs
