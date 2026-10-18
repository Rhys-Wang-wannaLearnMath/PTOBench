"""Official Fixture adapter for qwen3_14b.qwen3_14b_decode_tq_single_layer."""

import sys
from types import ModuleType, SimpleNamespace

import common


def _historical_config(official, identity):
    # Postponed dataclass annotations in the preserved config require a module
    # registration during execution. The shared loader isolates constants.py;
    # this temporary registration is confined to historical Fixture loading.
    previous = sys.modules.get(identity)
    sys.modules[identity] = ModuleType(identity)
    try:
        return common.load_isolated_module(
            official / "original_config.py", identity,
        )
    finally:
        if previous is None:
            sys.modules.pop(identity, None)
        else:
            sys.modules[identity] = previous


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(
        case_dir / case["config"], f"{identity}_public_config",
    )
    official = sealed / "official"
    historical = _historical_config(official, f"{identity}_original_config")
    alias = ModuleType("config")
    # Only the sealed historical adapter needs tuning and dynamic aliases.
    # Public fixed values override all historical evaluation values.
    alias.__dict__.update(vars(historical))
    alias.__dict__.update(vars(config))
    alias.QWEN3_14B = historical._Qwen3CompatModel(config.CASE_CONFIG)
    alias.QWEN3_14B_TILING = SimpleNamespace(
        seq_tile=config.BLOCK_SIZE, block_size=config.BLOCK_SIZE,
    )
    alias.BATCH = config.EVAL_BATCH
    alias.MAX_SEQ = config.EVAL_MAX_SEQ
    alias.NUM_LAYERS = config.EVAL_NUM_LAYERS
    # Keep MAX_BLOCKS_PER_SEQ at the source scratch capacity (32), independent
    # of the Fixture's physical single context block.
    # The shared loader validates and restores the complete isolated helper
    # scope before returning. Returned specs remain usable after that cleanup.
    return common.load_official_fixture_specs(
        official / "original.py", f"{identity}_official", seed,
        config_alias=alias,
    )
