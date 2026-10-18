"""Thin adapter for the byte-exact official V3.2 decode-front builder."""

import common

def build(case_dir, sealed, case, seed):
    del case_dir, case
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py", "decode_front_official", seed
    )
