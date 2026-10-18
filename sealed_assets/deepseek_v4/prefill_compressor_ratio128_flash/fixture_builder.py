"""Official Fixture adapter for deepseek_v4.prefill_compressor_ratio128_flash."""

from functools import partial
import sys
from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], f"{identity}_fixture_config",
    )
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    official_path = sealed / "official" / "original.py"
    rope_path = sealed / "official" / "rope_tables.py"
    previous_rope = sys.modules.get("rope_tables")
    original_loader = common.load_module

    def load_with_fixed_start(path, name, search_paths=()):
        module = original_loader(path, name, search_paths)
        if path.resolve() == official_path.resolve():
            module.build_tensor_specs = partial(
                module.build_tensor_specs, start_pos=config.START_POS,
            )
        return module

    sys.modules["rope_tables"] = original_loader(
        rope_path, f"{identity}_official_rope_tables",
    )
    common.load_module = load_with_fixed_start
    try:
        return common.load_official_fixture_specs(
            official_path, f"{identity}_official", seed, config_alias=alias,
        )
    finally:
        common.load_module = original_loader
        if previous_rope is None:
            sys.modules.pop("rope_tables", None)
        else:
            sys.modules["rope_tables"] = previous_rope
