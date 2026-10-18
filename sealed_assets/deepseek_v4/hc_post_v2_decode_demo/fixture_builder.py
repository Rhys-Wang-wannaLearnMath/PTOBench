"""Adapter around the byte-exact official TensorSpec builder."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "hc_post_v2_decode_demo_fixture_config")
    official_config = common.load_module(
        sealed / "official" / "original_config.py",
        "hc_post_v2_decode_demo_official_config",
    )
    alias = type(sys)("config")
    alias.__dict__.update(vars(official_config))
    alias.FLASH = config.CASE_CONFIG
    alias.DECODE_BATCH = config.DECODE_BATCH
    alias.DECODE_SEQ = config.DECODE_SEQ
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
        return common.load_official_fixture_specs(official_path, "hc_post_v2_decode_demo_official", seed, config_alias=alias)
    finally:
        common.load_module = original_loader
