"""Adapter around the byte-exact Qwen3-14B token-embedding Fixture builder."""

import importlib.util
import sys

import common


def build(case_dir, sealed, case, seed):
    config_path = case_dir / case["config"]
    config_name = "qwen3_14b_token_embed_config"
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Case config: {config_path}")
    config = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config
    try:
        spec.loader.exec_module(config)
    finally:
        sys.modules.pop(config_name, None)
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    return common.load_official_fixture_specs(
        sealed / "official" / "original.py",
        "qwen3_14b_token_embed_official",
        seed,
        config_alias=alias,
    )
