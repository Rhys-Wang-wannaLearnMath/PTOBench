"""Official fixture adapter for the byte-exact legacy SWA source."""

import importlib.util
from pathlib import Path

import common


def build(case_dir, sealed, case, seed):
    source = sealed / "official" / "original.py"
    helper_path = sealed / "official" / "deepseek_v4_decode_sparse_attn.py"
    original_spec = importlib.util.spec_from_file_location

    def case_local_spec(name, location, *args, **kwargs):
        if (
            name == "_deepseek_v4_sparse_attn_with_proj_compat"
            and Path(location).as_posix().endswith(
                "sources/models/deepseek/v4/deepseek_v4_decode_sparse_attn.py"
            )
        ):
            location = helper_path
        return original_spec(name, location, *args, **kwargs)

    importlib.util.spec_from_file_location = case_local_spec
    try:
        return common.load_official_fixture_specs(
            source,
            "deepseek_v4_decode_swa_official",
            seed,
        )
    finally:
        importlib.util.spec_from_file_location = original_spec
