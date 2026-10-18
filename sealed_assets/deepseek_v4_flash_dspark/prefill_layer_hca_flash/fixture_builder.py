"""Official Fixture adapter for deepseek_v4_flash_dspark.prefill_layer_hca_flash; no runtime execution."""
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

import common

def build(case_dir, sealed, case, seed):
    """Return official specs; fixed inputs come from this Case's public config."""
    identity = case["case_id"].replace(".", "_")
    public = common.load_isolated_module(
        Path(case_dir) / case["config"], identity + "_fixture_public_config",
    )
    if case["case_id"] != public.CASE_ID:
        raise ValueError("Fixture Case identity differs from the public contract")
    official_root = Path(sealed) / "official"
    config_alias = common.load_isolated_module(
        official_root / "original_config.py", identity + "_fixture_historical_config",
    )
    for name, value in vars(public).items():
        if not name.startswith("__"):
            setattr(config_alias, name, value)
    # Historical moe.py performs the expert reduction once after this raw binding.
    config_alias.FLASH = public.CASE_CONFIG
    config_alias.EP = public.SOURCE_EP
    config_alias.MOE_TOKENS = public.MOE_TOKENS
    config_alias.KV_ORI_MAX_BLOCKS = public.KV_ORI_MAX_BLOCKS
    config_alias.KV_CMP_MAX_BLOCKS = public.KV_CMP_MAX_BLOCKS
    config_alias.IDX_CACHE_MAX_BLOCKS = public.IDX_CACHE_MAX_BLOCKS
    previous_argv = sys.argv
    sys.argv = [identity + "_fixture", "--ep", str(public.EP), "--tp", str(public.TP_SIZE)]
    try:
        scope = nullcontext()
        with scope:
            return common.load_official_fixture_specs(
                official_root / "original.py", identity + "_fixture_official", seed,
                config_alias=config_alias, builder_name="build_tensor_specs",
                builder_args=(), builder_kwargs={"layer_id": public.LAYER_ID},
            )
    finally:
        sys.argv = previous_argv
