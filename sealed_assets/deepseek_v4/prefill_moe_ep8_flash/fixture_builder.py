"""Build FLASH EP8 prefill specs through the byte-exact official builder."""

import sys
import types

import common


def _load_specs(official, name, seed, alias, config):
    local_names = {path.stem for path in official.glob("*.py")}
    matches = lambda module_name: any(
        module_name == local or module_name.startswith(f"{local}.")
        for local in local_names
    )
    previous = {key: value for key, value in tuple(sys.modules.items()) if matches(key)}
    for key in previous:
        del sys.modules[key]
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0] if old_argv else "fixture_builder", "--ep", str(config.EP_WORLD_SIZE)]
    try:
        return common.load_official_fixture_specs(
            official / "original.py", name, seed, config_alias=alias,
            builder_kwargs={"layer_id": config.LAYER_ID, "num_tokens": config.NUM_TOKENS},
        )
    finally:
        sys.argv = old_argv
        for key in tuple(sys.modules):
            if matches(key):
                del sys.modules[key]
        sys.modules.update(previous)


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "deepseek_v4_prefill_moe_ep8_flash_public_config"
    )
    alias = types.ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return _load_specs(
        sealed / "official", "deepseek_v4_prefill_moe_ep8_flash_official", seed, alias, config
    )
