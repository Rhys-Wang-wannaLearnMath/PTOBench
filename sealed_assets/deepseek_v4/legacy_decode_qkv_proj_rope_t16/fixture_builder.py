"""Thin adapter around the byte-exact legacy official TensorSpec builder."""

import common


def build(case_dir, sealed, case, seed):
    del case_dir, case
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "legacy_decode_qkv_proj_rope_t16_official",
        seed,
    )
