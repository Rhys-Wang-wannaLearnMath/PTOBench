'Build prefill_layer_ep2_csa_flash fixtures through its byte-exact selected official closure.'

import builtins
import sys
import types
from pathlib import Path

import common
import golden


_CHILD_MUTABLE_NAMES = {
    "cmp_kv_state",
    "cmp_score_state",
    "inner_kv_state",
    "inner_score_state",
    "kv_cache",
    "cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
}



def _memoize_fixed_tables(specs):
    """Memoize only the official zero-argument deterministic state mappings.

    The tables contain no random draws. Cloning each return preserves their
    independent value semantics and all random initialization order.
    """
    replacements = {}
    names = {"init_compress_state_block_table", "init_inner_compress_state_block_table"}

    def visit(fn):
        if not isinstance(fn, types.FunctionType):
            return fn
        if fn in replacements:
            return replacements[fn]
        if fn.__name__ in names:
            cached = []
            def memoized():
                if not cached:
                    cached.append(fn())
                return cached[0].clone()
            replacements[fn] = memoized
            return memoized
        replacements[fn] = fn
        for cell in fn.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, types.FunctionType):
                cell.cell_contents = visit(value)
        return fn

    for spec in specs:
        if callable(getattr(spec, "init_value", None)):
            spec.init_value = visit(spec.init_value)
    return specs


def build(case_dir, sealed, case, seed):
    config = common.load_module(
        case_dir / case["config"],
        "deepseek_v4_prefill_layer_ep2_csa_flash_fixture_config",
    )
    alias = types.ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG

    official_root = (sealed / "official").resolve()
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

    original_argv = sys.argv
    original_loader = common.load_module
    tensor_spec_type = golden.TensorSpec

    def official_isinstance(value, class_or_tuple):
        if class_or_tuple is golden.TensorSpec:
            return builtins.isinstance(value, tensor_spec_type)
        return builtins.isinstance(value, class_or_tuple)

    def with_legacy_directions(builder):
        def wrapped(*args, **kwargs):
            specs = _memoize_fixed_tables(builder(*args, **kwargs))
            for spec in specs:
                if not builtins.isinstance(spec, tensor_spec_type):
                    continue
                if spec.name in _CHILD_MUTABLE_NAMES:
                    spec.direction = "inout"
                elif spec.name == "x_out":
                    spec.direction = "out"
                else:
                    spec.direction = "in"
            return specs

        return wrapped

    def load_with_fixed_invocation(path, name, search_paths=()):
        if path.resolve() == official_path:
            # Builder-delayed imports run after load_module's scoped search
            # path is restored. Keep the owning official module available
            # until construction completes, with no Case sys.path changes.
            sys.modules["rope_tables"] = original_loader(
                official_root / "rope_tables.py", "rope_tables", (official_root,),
            )
        module = original_loader(path, name, search_paths)
        if path.resolve() == official_path:
            module.isinstance = official_isinstance
            for local_name in local_names:
                local_module = sys.modules.get(local_name)
                if local_module is None:
                    continue
                local_module.isinstance = official_isinstance
                module_path = getattr(local_module, "__file__", None)
                if module_path is not None and not Path(module_path).resolve().is_relative_to(official_root):
                    raise ImportError(
                        f"official dependency {local_name!r} resolved outside {official_root}: {module_path}"
                    )
            for builder_name in (
                "build_swa_attention_tensor_specs",
                "build_hca_attention_tensor_specs",
                "build_csa_attention_tensor_specs",
            ):
                setattr(module, builder_name, with_legacy_directions(getattr(module, builder_name)))
            module._KIND_BUILDER = {key: with_legacy_directions(value)
                                    for key, value in module._KIND_BUILDER.items()}
            builder = module.build_tensor_specs
            module.build_tensor_specs = lambda: builder(
                layer_id=config.LAYER_ID,
                chunk_lens=config.CHUNK_LENS,
                start_positions=config.START_POSITIONS,
            )
        return module

    sys.argv = [original_argv[0] if original_argv else "fixture_builder", "--ep", str(config.EP_WORLD_SIZE)]
    common.load_module = load_with_fixed_invocation
    try:
        specs = common.load_official_fixture_specs(
            official_path,
            "deepseek_v4_prefill_layer_ep2_csa_flash_official",
            seed,
            config_alias=alias,
        )
        for spec in specs:
            if not builtins.isinstance(spec, tensor_spec_type):
                continue
            if spec.name in case["state"]["initial_state"]:
                spec.direction = "inout"
            elif spec.name in case["correctness"]["outputs"]:
                spec.direction = "out"
            else:
                spec.direction = "in"
        return specs
    finally:
        common.load_module = original_loader
        sys.argv = original_argv
        for name in tuple(sys.modules):
            if matches(name):
                del sys.modules[name]
        sys.modules.update(previous_modules)
