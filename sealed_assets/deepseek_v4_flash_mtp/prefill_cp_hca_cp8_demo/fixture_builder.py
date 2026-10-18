"""Official Fixture adapter for the DEMO CP8 HCA prefill Case."""

import sys
from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], f"{identity}_fixture_config")
    config_alias = ModuleType("config")
    config_alias.__dict__.update(vars(config))
    config_alias.FLASH = config.CASE_CONFIG
    official_path = sealed / "official" / "original.py"
    previous_argv = sys.argv
    sys.argv = [str(official_path), "--cp", str(config.CP_SIZE)]
    try:
        return common.load_official_fixture_specs(
            official_path, f"{identity}_official", seed, config_alias=config_alias
        )
    finally:
        sys.argv = previous_argv
