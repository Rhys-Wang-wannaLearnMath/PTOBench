"""Official Fixture adapter for qwen3_14b.prefill_fwd."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "qwen3_14b_prefill_fwd_fixture_config"
    )
    official_root = sealed / "official"
    # Preserve the historical dependency's spelling without changing its bytes
    # or substituting model MAX_SEQ/NUM_LAYERS with the Fixture's explicit args.
    with common._isolated_module_scope((official_root, case_dir / "public")):
        sys.modules["config"] = common.load_module(
            official_root / "original_config.py",
            "qwen3_14b_prefill_fwd_original_config",
        )
        return common.load_official_fixture_specs(
            official_root / "original.py",
            "qwen3_14b_prefill_fwd_official",
            seed,
            builder_args=config.FIXTURE_BUILDER_ARGS,
        )
