"""Adapter for the byte-exact official ratio-0 TensorSpec builder."""

import common


def build(case_dir, sealed, case, seed):
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "deepseek_v4_decode_compressor_ratio0_official",
        seed,
    )
