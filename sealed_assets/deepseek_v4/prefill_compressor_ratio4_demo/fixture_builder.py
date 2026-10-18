"""Official Fixture adapter for deepseek_v4.prefill_compressor_ratio4_demo."""

from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], f"{identity}_fixture_config")
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    # The byte-exact source imports FLASH, including for DEMO and PRO bindings.
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py", f"{identity}_official", seed,
        config_alias=alias, builder_kwargs={"start_pos": config.START_POS},
    )
