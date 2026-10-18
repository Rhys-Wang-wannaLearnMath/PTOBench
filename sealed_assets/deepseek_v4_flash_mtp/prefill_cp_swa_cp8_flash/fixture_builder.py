"""Official Fixture adapter for deepseek_v4_flash_mtp.prefill_cp_swa_cp8_flash."""

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
        specs, _context = common.load_official_fixture_specs(
            official_path, f"{identity}_official", seed, config_alias=config_alias,
            builder_kwargs={"cp_size": config.CP_SIZE},
        )
        # The official builder has already materialized inputs in source RNG
        # order. Only its owner-rank spec position differs from the host ABI.
        specs = list(specs)
        names = [spec.name for spec in specs]
        owner_rank = specs.pop(names.index("owner_rank_table"))
        specs.insert(names.index("final_win_seg_src"), owner_rank)
        return specs
    finally:
        sys.argv = previous_argv
