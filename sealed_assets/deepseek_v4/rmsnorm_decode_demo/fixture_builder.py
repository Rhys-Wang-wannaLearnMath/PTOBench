"""Adapter around the byte-exact official RMSNorm TensorSpec builder."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "deepseek_v4_rmsnorm_decode_demo_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    alias.PREFILL_BATCH = 1
    alias.PREFILL_SEQ = 128
    official_path = sealed / "official" / "original.py"
    original_loader = common.load_module

    def load_with_decode_args(path, name, search_paths=()):
        module = original_loader(path, name, search_paths)
        if path.resolve() == official_path.resolve():
            builder = module.build_tensor_specs
            module.build_tensor_specs = lambda: builder(config.DECODE_BATCH, config.DECODE_SEQ)
        return module

    common.load_module = load_with_decode_args
    try:
        return common.load_official_fixture_specs(
            official_path, "deepseek_v4_rmsnorm_decode_demo_official", seed,
            config_alias=alias,
        )
    finally:
        common.load_module = original_loader
