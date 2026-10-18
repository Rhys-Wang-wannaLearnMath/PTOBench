"""Official TensorSpec fixture builder for the FLASH combine Case."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "combine_flash_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))

    class _Preset:
        hidden_size = config.D
        num_experts_per_tok = config.TOPK
        n_routed_experts = config.N_ROUTED_EXPERTS

    alias.FLASH = _Preset()
    alias.DECODE_BATCH = config.B
    alias.DECODE_SEQ = config.S
    alias.EP_WORLD_SIZE = config.EP_WORLD_SIZE
    alias.RECV_MAX = config.RECV_MAX
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "combine_flash_official",
        seed,
        config_alias=alias,
    )
