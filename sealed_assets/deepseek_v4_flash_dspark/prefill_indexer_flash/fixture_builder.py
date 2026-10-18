"""Bind the selected fixed contract to its byte-exact official Fixture builder."""

from pathlib import Path
import types

import common


def build(case_dir, sealed, case, seed):
    case_dir, sealed = Path(case_dir), Path(sealed)
    case_id = case["case_id"]
    identity = case_id.replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], f"fixture_{identity}_config"
    )
    if config.CASE_ID != case_id:
        raise ValueError("public config identity mismatch")
    expected_seed = common.read_json(sealed / "fixture_policy.json")[
        "fixture_seeds"
    ]["fixture_000"]
    if seed != expected_seed:
        raise ValueError("fixture seed mismatch")
    if case["scalars"] != ["num_tokens"]:
        raise ValueError("scalar contract mismatch")

    original_config = common.load_module(
        sealed / "official" / "original_config.py",
        f"fixture_{identity}_original_config",
    )
    alias = types.ModuleType(f"fixture_{identity}_config_alias")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    # Inert imports of unused historical utility branches stay sealed. Every
    # indexer binding, including the ratio-4 state page size, comes from public.
    for name in (
        "DECODE_BATCH", "DECODE_SEQ", "DECODE_START_POS",
        "C128_COMPRESSOR_BLOCK_SIZE",
    ):
        setattr(alias, name, getattr(original_config, name))

    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        f"fixture_{identity}_official",
        seed,
        config_alias=alias,
        builder_kwargs={
            "start_pos": config.START_POS,
            "num_tokens": config.NUM_TOKENS,
        },
    )
