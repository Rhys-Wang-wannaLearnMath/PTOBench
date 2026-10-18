"""Official Fixture adapter for the byte-exact DSpark projection builder."""
from pathlib import Path
from types import ModuleType
import common

def build(case_dir, sealed, case, seed):
    case_dir, sealed = Path(case_dir), Path(sealed)
    policy = common.read_json(sealed / "fixture_policy.json")
    if seed != policy["fixture_seeds"]["fixture_000"]:
        raise ValueError("fixture_000 seed mismatch")
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], f"fixture_{identity}_config")
    original_config = common.load_module(sealed / "official" / "original_config.py", f"fixture_{identity}_original_config")
    alias = ModuleType(f"fixture_{identity}_alias")
    alias.__dict__.update(vars(config)); alias.FLASH = config.CASE_CONFIG
    for name in ("DECODE_BATCH", "DECODE_SEQ", "PREFILL_BATCH", "PREFILL_SEQ", "TP"):
        setattr(alias, name, getattr(config, name, getattr(original_config, name)))
    return common.load_official_fixture_specs(sealed / "official" / "original.py", f"fixture_{identity}_official", seed, config_alias=alias)
