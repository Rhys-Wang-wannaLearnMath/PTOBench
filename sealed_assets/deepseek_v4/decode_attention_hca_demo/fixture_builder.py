"""Official TensorSpec adapter for the DEMO ratio-128 HCA contract."""

import sys

import common
import torch


def _load_specs(official, name, seed, alias):
    """Load the frozen multi-file builder without sibling module reuse."""
    local_names = {path.stem for path in official.glob("*.py")}
    local_names.add("config")
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
    config = common.load_module(case_dir / case["config"], "decode_attention_hca_demo_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    specs = _load_specs(
        sealed / "official",
        "decode_attention_hca_demo_official",
        seed,
        alias,
    )
    # The frozen builder's default is homogeneous; the approved fixture binds
    # its explicit heterogeneous (128, 127, ...) start-position arguments.
    start = torch.tensor([config.START_POS, max(config.START_POS - 1, 0)], dtype=torch.int32)
    start = start.repeat((config.B + start.numel() - 1) // start.numel())[:config.B].clone()
    seq = start.to(torch.int64) + config.S
    used = torch.where(seq <= config.WIN, seq, config.WIN + seq // config.COMPRESS_RATIO).to(torch.int32)
    for spec in specs:
        if spec.name == "start_pos":
            spec.init_value = start
        elif spec.name == "seqused_kv":
            spec.init_value = used
    return specs
