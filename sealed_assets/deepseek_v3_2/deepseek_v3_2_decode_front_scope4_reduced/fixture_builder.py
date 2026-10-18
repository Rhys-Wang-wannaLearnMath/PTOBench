"""Build reduced-profile specs through the byte-exact official initializer."""

import common


def _load_specs(path, name, seed, builder_args):
    original_loader = common.load_module

    def load_profile_module(module_path, module_name, search_paths=()):
        module = original_loader(module_path, module_name, search_paths)
        build_tensor_specs = module.build_tensor_specs
        module.build_tensor_specs = lambda: build_tensor_specs(*builder_args)
        return module

    common.load_module = load_profile_module
    try:
        return common.load_official_fixture_specs(path, name, seed)
    finally:
        common.load_module = original_loader


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "scope4_reduced_fixture_config")
    return _load_specs(
        sealed / "official" / "original.py", "scope4_reduced_official", seed,
        config.CASE_CONFIG.builder_args,
    )
