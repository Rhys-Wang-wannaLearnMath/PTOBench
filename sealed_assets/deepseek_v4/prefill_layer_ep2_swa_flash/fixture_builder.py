"""Build the FLASH EP2 SWA Fixture through the byte-exact official closure."""

import builtins
import sys
import types

import common
import golden


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"],
        "deepseek_v4_prefill_layer_ep2_swa_flash_fixture_config",
    )
    alias = types.ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG

    official_root = sealed / "official"
    official_path = official_root / "original.py"
    local_names = {path.stem for path in official_root.glob("*.py")}

    def matches(module_name):
        return any(
            module_name == local or module_name.startswith(f"{local}.")
            for local in local_names
        )

    previous_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if matches(name)
    }
    for name in previous_modules:
        del sys.modules[name]

    original_argv = sys.argv[:]
    original_loader = common.load_module
    official_search_root = str(official_root)
    tensor_spec_type = golden.TensorSpec

    def official_isinstance(value, class_or_tuple):
        if class_or_tuple is golden.TensorSpec:
            return builtins.isinstance(value, tensor_spec_type)
        return builtins.isinstance(value, class_or_tuple)

    def with_legacy_directions(builder):
        def wrapped(*args, **kwargs):
            specs = builder(*args, **kwargs)
            for spec in specs:
                if builtins.isinstance(spec, tensor_spec_type):
                    spec.direction = "out" if spec.name in {"kv_cache", "x_out"} else "in"
            return specs

        return wrapped

    def load_with_fixed_invocation(path, name, search_paths=()):
        module = original_loader(path, name, search_paths)
        if path.resolve() == official_path.resolve():
            module.isinstance = official_isinstance
            for local_name in local_names:
                local_module = sys.modules.get(local_name)
                if local_module is not None:
                    local_module.isinstance = official_isinstance
            for builder_name in (
                "build_swa_attention_tensor_specs",
                "build_hca_attention_tensor_specs",
                "build_csa_attention_tensor_specs",
            ):
                setattr(module, builder_name, with_legacy_directions(getattr(module, builder_name)))
            builder = module.build_tensor_specs
            module.build_tensor_specs = lambda: builder(
                layer_id=config.LAYER_ID,
                chunk_lens=config.CHUNK_LENS,
                start_positions=config.START_POSITIONS,
            )
        return module

    sys.argv = [original_argv[0] if original_argv else "fixture_builder", "--ep", "2"]
    sys.path.insert(0, official_search_root)
    common.load_module = load_with_fixed_invocation
    try:
        return common.load_official_fixture_specs(
            official_path,
            "deepseek_v4_prefill_layer_ep2_swa_flash_official",
            seed,
            config_alias=alias,
        )
    finally:
        common.load_module = original_loader
        sys.argv = original_argv
        del sys.path[0]
        for name in tuple(sys.modules):
            if matches(name):
                del sys.modules[name]
        sys.modules.update(previous_modules)
