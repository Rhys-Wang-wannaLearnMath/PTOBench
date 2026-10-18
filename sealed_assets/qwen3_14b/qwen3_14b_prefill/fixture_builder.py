"""Adapter around the byte-exact official Qwen3 prefill Fixture builder."""

import common


def build(case_dir, sealed, case, seed):
    del case_dir, case
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "qwen3_14b_prefill_official",
        seed,
    )
