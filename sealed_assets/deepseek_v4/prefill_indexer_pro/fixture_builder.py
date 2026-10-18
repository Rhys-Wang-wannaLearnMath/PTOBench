"""Official Fixture adapter for deepseek_v4.prefill_indexer_pro."""

import sys

import common


def _load_specs(official, name, seed, alias):
    local_names = {path.stem for path in official.glob("*.py")} | {"config"}
    matches = lambda module_name: any(
        module_name == local or module_name.startswith(f"{local}.")
        for local in local_names
    )
    previous = {
        module_name: module
        for module_name, module in tuple(sys.modules.items())
        if matches(module_name)
    }
    for module_name in previous:
        del sys.modules[module_name]
    try:
        sys.modules["rope_tables"] = common.load_module(
            official / "rope_tables.py", f"{name}_rope_tables"
        )
        return common.load_official_fixture_specs(
            official / "original.py", name, seed, config_alias=alias
        )
    finally:
        for module_name in tuple(sys.modules):
            if matches(module_name):
                del sys.modules[module_name]
        sys.modules.update(previous)


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "deepseek_v4_prefill_indexer_pro_fixture_config"
    )
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return _load_specs(
        sealed / "official", "deepseek_v4_prefill_indexer_pro_official", seed, alias
    )
