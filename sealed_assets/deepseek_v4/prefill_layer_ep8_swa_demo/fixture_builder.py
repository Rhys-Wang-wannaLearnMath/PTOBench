"""Official Fixture adapter for deepseek_v4.prefill_layer_ep8_swa_demo."""

import ast
import sys
import types

import common
import golden


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    alias = types.ModuleType("config")
    alias.__dict__.update(vars(config))
    # The byte-exact historical closure imports FLASH for either selected preset.
    alias.FLASH = config.CASE_CONFIG
    original = sealed / "official" / "original.py"
    order = next(
        ast.literal_eval(node.value)
        for node in ast.parse(original.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "HOST_TENSOR_ORDER"
                for target in node.targets)
    )
    previous_argv = sys.argv
    try:
        sys.argv = [identity + "_fixture", "--ep", str(config.EP_WORLD_SIZE)]
        specs = common.load_official_fixture_specs(
            original, identity + "_official_fixture", seed,
            config_alias=alias,
            builder_kwargs={"layer_id": config.LAYER_ID,
                            "chunk_lens": config.CHUNK_LENS,
                            "start_positions": config.START_POSITIONS},
        )
    finally:
        sys.argv = previous_argv
    if [spec.name for spec in specs] != [*order, *case["scalars"]]:
        raise ValueError("official Fixture ABI differs from fixed Case")
    for spec in specs:
        if isinstance(spec, golden.TensorSpec):
            spec.direction = ("inout" if spec.name in case["state"]["initial_state"]
                              else "out" if spec.name in case["correctness"]["outputs"]
                              else "in")
    return specs
