"""Adapter around the byte-exact official TensorSpec builder."""

import common


def build(case_dir, sealed, case, seed):
    del case_dir, case
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        f"qwen3_14b_decode_scope2_official_{seed}",
        seed,
    )
