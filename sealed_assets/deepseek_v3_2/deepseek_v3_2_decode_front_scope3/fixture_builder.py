"""Adapter around the byte-exact official TensorSpec builder."""

import common


def build(case_dir, sealed, case, seed):
    del case_dir, case
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "deepseek_v3_2_decode_front_scope3_official",
        seed,
    )
