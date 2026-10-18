"""Thin adapter for the byte-exact historical Qwen3 decode builder."""

import os
import sys

import common


def build(case_dir, sealed, case, seed):
    """Build official specs with full_seq=False and the Case public config alias."""
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"],
                                f"{identity}_contract_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    original_loader = common.load_module
    previous_max_seq = os.environ.get("PTO2_MANUAL_MAX_SEQ")

    def load_builder(path, name, search_paths=()):
        module = original_loader(path, name, search_paths)
        if path.resolve() == (sealed / "official" / "original.py").resolve():
            module.build_tensor_specs = lambda: module._build_specs(
                module.random_inputs(full_seq=config.FULL_SEQ, seed=seed)
            )
        return module

    common.load_module = load_builder
    os.environ["PTO2_MANUAL_MAX_SEQ"] = str(config.CASE_CONFIG.max_seq)
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            f"{identity}_official",
            seed,
            config_alias=alias,
        )
    finally:
        common.load_module = original_loader
        if previous_max_seq is None:
            os.environ.pop("PTO2_MANUAL_MAX_SEQ", None)
        else:
            os.environ["PTO2_MANUAL_MAX_SEQ"] = previous_max_seq
