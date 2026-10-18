"""Official Fixture adapter for deepseek_v4_flash_mtp.prefill_cp_hca_cp4_demo (DEMO CP4 HCA)."""

import sys
from types import ModuleType

import torch
import common


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    assert case["case_id"] == "deepseek_v4_flash_mtp.prefill_cp_hca_cp4_demo"
    assert config.CASE_CONFIG.name == "demo"
    root = sealed / "official"
    historical = common.load_module(root / "original_config.py", identity + "_original_config")
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    # These names serve only unused historical DSL imports/defaults. Every live
    # HCA Fixture dimension, capacity, quantization value and preset is public.
    for name in (
        "DECODE_BATCH", "DECODE_SEQ", "CSA_STATE_PHYSICAL_BLOCKS",
        "CSA_INNER_STATE_PHYSICAL_BLOCKS", "PREFILL_IDX_BLOCK_NUM",
        "PREFILL_IDX_MAX_BLOCKS", "PREFILL_ORI_BLOCK_NUM",
    ):
        setattr(alias, name, getattr(historical, name))

    previous_argv = sys.argv
    try:
        sys.argv = [str(root / "original.py"), "--cp", str(config.CP_SIZE)]

        with torch.random.fork_rng(devices=[]):
            return common.load_official_fixture_specs(
                root / "original.py", identity + "_official", seed,
                config_alias=alias, builder_args=(config.CP_SIZE,),
            )
    finally:

        sys.argv = previous_argv
