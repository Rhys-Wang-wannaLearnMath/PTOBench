"""Official Fixture adapter for deepseek_v4.decode_indexer_flash."""

from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "decode_indexer_flash_fixture_config"
    )
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    # The byte-exact source imports FLASH; bind this Case's selected preset.
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py", "decode_indexer_flash_official", seed,
        config_alias=alias, builder_args=(config.START_POS, True),
    )
