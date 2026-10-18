"""Adapter around the byte-exact official TensorSpec builder."""

import common


def build(case_dir, sealed, case, seed):
    del case_dir, case
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "decode_moe_expert_official",
        seed,
    )
