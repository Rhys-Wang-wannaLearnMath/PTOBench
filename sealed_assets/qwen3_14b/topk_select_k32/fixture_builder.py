"""Adapter for the byte-exact historical Qwen3 top-k Fixture builder."""

from __future__ import annotations

import importlib.util
import sys
import types

import common


def _public_config(case_dir, case):
    path = case_dir / case["config"]
    name = "qwen3_14b_topk_select_k32_public_config"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Case config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def build(case_dir, sealed, case, seed):
    """Build historical TensorSpecs with the fixed selection_k=32 binding."""
    config = _public_config(case_dir, case)
    official_dir = sealed / "official"
    original_config = common.load_isolated_module(
        official_dir / "original_config.py",
        "qwen3_14b_topk_select_k32_official_config",
        (official_dir,),
    )
    alias = types.ModuleType("config")
    alias.QWEN3_14B = config.CASE_CONFIG
    alias.QWEN3_14B_TILING = original_config.QWEN3_14B_TILING

    original_loader = common.load_module

    def load_and_bind(path, name, search_paths=()):
        module = original_loader(path, name, search_paths)
        if path.resolve() == (official_dir / "original.py").resolve():
            historical_builder = module.build_tensor_specs
            module.build_tensor_specs = lambda: historical_builder(
                selection_k=config.SELECTION_K
            )
        return module

    common.load_module = load_and_bind
    try:
        return common.load_official_fixture_specs(
            official_dir / "original.py",
            "qwen3_14b_topk_select_k32_official",
            seed,
            config_alias=alias,
        )
    finally:
        common.load_module = original_loader
