"""Adapter around the byte-exact official Qwen3-32B 4-D Fixture builder."""

import common


def build(case_dir, sealed, case, seed):
    del case_dir, case
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "qwen3_32b_decode_4d_official",
        seed,
    )
