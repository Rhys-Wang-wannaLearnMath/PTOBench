"""Official Fixture adapter for deepseek_v4.decode_indexer_compressor_pro."""

import sys

import common
import torch


def _load_specs(official, name, seed, alias):
    local_names = {path.stem for path in official.glob("*.py")}
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
        case_dir / case["config"], "decode_indexer_compressor_pro_fixture_config"
    )
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    # The byte-exact official source historically imports FLASH as M.
    alias.FLASH = config.CASE_CONFIG
    specs = _load_specs(
        sealed / "official", "decode_indexer_compressor_pro_official", seed, alias
    )
    # Bind the approved historical builder call (start_pos=3, hetero_start_pos=True).
    pattern = torch.tensor([0, config.COMPRESS_RATIO - config.DECODE_SEQ,
                            config.COMPRESS_RATIO - 1, config.COMPRESS_RATIO * 2 - 1],
                           dtype=torch.int32)
    start = pattern.repeat((config.B + pattern.numel() - 1) // pattern.numel())[:config.B]
    for spec in specs:
        if spec.name == "start_pos":
            spec.init_value = start.clone()
    return specs
