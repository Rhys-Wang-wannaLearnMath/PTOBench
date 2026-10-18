"""Official Fixture adapter for deepseek_v4.prefill_layer_ep8_hca_demo."""

import builtins
import importlib
import sys
import types
from pathlib import Path

import common
import golden


_CHILD_MUTABLE_NAMES = {
    "cmp_kv_state", "cmp_score_state", "inner_kv_state", "inner_score_state",
    "kv_cache", "cmp_kv", "idx_kv_cache", "idx_kv_scale",
}


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"], "deepseek_v4_prefill_layer_ep8_hca_demo_fixture_config",
    )
    alias = types.ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    official_root = (sealed / "official").resolve()
    official_path = official_root / "original.py"
    local_names = {path.stem for path in official_root.glob("*.py")}
    tensor_spec_type = golden.TensorSpec
    original_loader = common.load_module
    original_argv = sys.argv

    def official_isinstance(value, class_or_tuple):
        if class_or_tuple is golden.TensorSpec:
            return builtins.isinstance(value, tensor_spec_type)
        return builtins.isinstance(value, class_or_tuple)

    def stamp_child(builder):
        def wrapped(*args, **kwargs):
            specs = builder(*args, **kwargs)
            for spec in specs:
                if builtins.isinstance(spec, tensor_spec_type):
                    spec.direction = (
                        "inout" if spec.name in _CHILD_MUTABLE_NAMES
                        else "out" if spec.name == "x_out" else "in"
                    )
            return specs
        return wrapped

    def check_local_modules():
        for local_name in local_names:
            local_module = sys.modules.get(local_name)
            if local_module is None:
                continue
            module_path = getattr(local_module, "__file__", None)
            if module_path is None or not Path(module_path).resolve().is_relative_to(official_root):
                raise ImportError(f"official dependency {local_name!r} escaped {official_root}")
            local_module.isinstance = official_isinstance

    def load_fixed_official(path, name, search_paths=()):
        module = original_loader(path, name, search_paths)
        if path.resolve() != official_path:
            return module
        importlib.import_module("rope_tables")
        check_local_modules()
        module.isinstance = official_isinstance
        for kind in ("swa", "hca", "csa"):
            builder_name = f"build_{kind}_attention_tensor_specs"
            wrapped = stamp_child(getattr(module, builder_name))
            setattr(module, builder_name, wrapped)
            module._KIND_BUILDER[kind] = wrapped
        builder = module.build_tensor_specs

        def build_fixed():
            specs = builder(
                layer_id=config.LAYER_ID,
                chunk_lens=config.CHUNK_LENS,
                start_positions=config.START_POSITIONS,
            )
            check_local_modules()
            expected_names = [*module.HOST_TENSOR_ORDER, *case["scalars"]]
            if [spec.name for spec in specs] != expected_names:
                raise ValueError("official Fixture ABI differs from the fixed Case")
            return specs
        module.build_tensor_specs = build_fixed
        return module

    sys.argv = [
        original_argv[0] if original_argv else "deepseek_v4_prefill_layer_ep8_hca_demo_fixture",
        "--ep", str(config.EP_WORLD_SIZE),
    ]
    common.load_module = load_fixed_official
    try:
        specs = common.load_official_fixture_specs(
            official_path, "deepseek_v4_prefill_layer_ep8_hca_demo_official", seed, config_alias=alias,
        )
        for spec in specs:
            if builtins.isinstance(spec, tensor_spec_type):
                spec.direction = (
                    "inout" if spec.name in case["state"]["initial_state"]
                    else "out" if spec.name in case["correctness"]["outputs"] else "in"
                )
        return specs
    finally:
        common.load_module = original_loader
        sys.argv = original_argv
