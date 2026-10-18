"""Official TensorSpec adapter for the DEMO sparse-attention contract."""

import sys

import common
import torch


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(
        case_dir / case["config"],
        f"{identity}_fixture_config",
    )
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    torch_state = torch.random.get_rng_state()
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            f"{identity}_official",
            seed,
            config_alias=alias,
        )
    finally:
        torch.random.set_rng_state(torch_state)
