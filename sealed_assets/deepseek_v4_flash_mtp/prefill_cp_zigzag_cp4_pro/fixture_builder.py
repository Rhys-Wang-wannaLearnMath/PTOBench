"""Official Fixture adapter for the PRO CP4 zigzag exchange Case."""

import sys

import common


def build(case_dir, sealed, case, seed):
    result_id = case["case_id"].replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], f"{result_id}_fixture_config"
    )
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    official_path = sealed / "official" / "original.py"
    previous_argv = sys.argv
    sys.argv = [str(official_path), "--cp", str(config.CP_SIZE)]
    try:
        return common.load_official_fixture_specs(
            official_path, f"{result_id}_official", seed, config_alias=alias
        )
    finally:
        sys.argv = previous_argv
