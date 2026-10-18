"""Official Fixture adapter for deepseek_v4.prefill_moe_ep4_flash; never executes an NPU program."""

import os
import sys
from contextlib import contextmanager
from types import ModuleType

import common


@contextmanager
def _metadata_imports(official_root, config):
    previous_argv = sys.argv
    previous_hc_impl = os.environ.get("DSV4_HC_PRE_IMPL")
    try:
        sys.argv = [str(official_root / "original.py"), "--ep", str(config.EP_WORLD_SIZE)]
        os.environ["DSV4_HC_PRE_IMPL"] = "separate"
        yield
    finally:
        sys.argv = previous_argv
        if previous_hc_impl is None:
            os.environ.pop("DSV4_HC_PRE_IMPL", None)
        else:
            os.environ["DSV4_HC_PRE_IMPL"] = previous_hc_impl


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    # The byte-exact moe.py performs //8*EP itself. Bind the unscaled
    # selected object here so it applies that source expression exactly once.
    alias.FLASH = config.FLASH
    # Only historical helper annotation/standalone scaffolding uses decode.
    alias.DECODE_BATCH = 8
    alias.DECODE_SEQ = 1
    alias.DECODE_TOKENS = alias.DECODE_BATCH * alias.DECODE_SEQ
    official_root = sealed / "official"
    with _metadata_imports(official_root, config):
        return common.load_official_fixture_specs(
            official_root / "original.py", identity + "_official", seed,
            config_alias=alias,
            builder_args=(config.LAYER_ID, config.NUM_TOKENS),
        )
