"""Official TensorSpec adapter for the DEMO ratio-128 compressor contract."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "deepseek_v4_decode_compressor_ratio128_demo_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    load_module = common.load_module

    def load_with_fixed_invocation(path, name, search_paths=()):
        official = load_module(path, name, search_paths)
        builder = official.build_tensor_specs
        official.build_tensor_specs = lambda: builder(127, True, False, False)
        return official

    common.load_module = load_with_fixed_invocation
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            "deepseek_v4_decode_compressor_ratio128_demo_official",
            seed,
            config_alias=alias,
        )
    finally:
        common.load_module = load_module
