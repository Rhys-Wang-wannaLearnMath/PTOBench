"""Case-local adapter for the byte-exact official HCA Fixture builder."""
from __future__ import annotations
import importlib.util
import types
from pathlib import Path
import common


def _load_public_config(path: Path, case_id: str):
    name = f"fixture_{case_id.replace('.', '_')}_public_config"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load public config {path}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def build(case_dir, sealed, case, seed):
    case_dir = Path(case_dir)
    sealed = Path(sealed)
    case_id = case["case_id"]
    config = _load_public_config(case_dir / case["config"], case_id)
    if case_id != config.CASE_ID:
        raise ValueError(f"public config CASE_ID={config.CASE_ID} disagrees with {case_id}")
    policy = common.read_json(sealed / "fixture_policy.json")
    expected_seed = policy["fixture_seeds"]["fixture_000"]
    if seed != expected_seed:
        raise ValueError(f"fixture_000 requires seed {expected_seed}, got {seed}")
    identity = case_id.replace(".", "_")
    original_config = common.load_module(
        sealed / "official" / "original_config.py",
        f"fixture_{identity}_original_config",
    )
    config_alias = types.ModuleType(f"fixture_{identity}_config_alias")
    config_alias.__dict__.update(vars(config))
    config_alias.FLASH = config.CASE_CONFIG
    for name in ("DECODE_BATCH", "DECODE_SEQ", "DECODE_START_POS", "C4A_COMPRESSOR_BLOCK_SIZE"):
        setattr(config_alias, name, getattr(original_config, name))
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        f"fixture_{identity}_official",
        seed,
        config_alias=config_alias,
        builder_kwargs={"start_pos": config.START_POS, "num_tokens": config.NUM_TOKENS},
    )
