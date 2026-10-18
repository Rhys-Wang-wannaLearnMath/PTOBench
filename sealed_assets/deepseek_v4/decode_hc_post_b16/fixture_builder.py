"""Adapter around the byte-exact official TensorSpec builder."""

import common


def build(case_dir, sealed, case, seed):
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "decode_hc_post_b16_official",
        seed,
    )
