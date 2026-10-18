"""Adapter for the byte-exact official Scope 1 Fixture builder."""

import common


def build(case_dir, sealed, case, seed):
    del case_dir, case
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "qwen3_32b_decode_scope1_official",
        seed,
    )
