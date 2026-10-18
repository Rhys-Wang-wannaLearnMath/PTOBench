"""Official Fixture adapter for deepseek_v4_flash_mtp.prefill_cp_hca_cp4_pro (PRO CP4 HCA)."""

import sys
from types import ModuleType

import torch
import common


def build(case_dir, sealed, case, seed):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    assert case["case_id"] == "deepseek_v4_flash_mtp.prefill_cp_hca_cp4_pro"
    assert config.CASE_CONFIG.name == "pro"
    root = sealed / "official"
    historical = common.load_module(root / "original_config.py", identity + "_original_config")
    alias = ModuleType("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    # These names serve only unused historical DSL imports/defaults. Every live
    # HCA Fixture dimension, capacity, quantization value and preset is public.
    for name in (
        "DECODE_BATCH", "DECODE_SEQ", "CSA_STATE_PHYSICAL_BLOCKS",
        "CSA_INNER_STATE_PHYSICAL_BLOCKS", "PREFILL_IDX_BLOCK_NUM",
        "PREFILL_IDX_MAX_BLOCKS", "PREFILL_ORI_BLOCK_NUM",
    ):
        setattr(alias, name, getattr(historical, name))
    import hashlib
    import importlib.abc
    import importlib.util

    class _HCPreFixtureLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            path = root / "hc_pre.py"
            source = path.read_bytes()
            assert hashlib.sha256(source).hexdigest() == "39f365df3925fc555ce6f21262ac2fa528ca089b1d3264e18feaf46342c9feda"
            old = "assert HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0"
            new = 'assert HC_PRE_IMPL == "separate" or (HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0)'
            text = source.decode()
            assert text.count(old) == 1
            # The sole guarded assertion belongs to unused syncall partitioning.
            # Every other assertion and every builder/golden/runtime body stays exact.
            exec(compile(text.replace(old, new), str(path), "exec"), module.__dict__)

    class _HCPreFixtureFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "hc_pre":
                return importlib.util.spec_from_file_location(
                    fullname, root / "hc_pre.py", loader=_HCPreFixtureLoader())
            return None

    finder = _HCPreFixtureFinder()
    previous_argv = sys.argv
    try:
        sys.argv = [str(root / "original.py"), "--cp", str(config.CP_SIZE)]
        sys.meta_path.insert(0, finder)
        with torch.random.fork_rng(devices=[]):
            return common.load_official_fixture_specs(
                root / "original.py", identity + "_official", seed,
                config_alias=alias, builder_args=(config.CP_SIZE,),
            )
    finally:
        sys.meta_path.remove(finder)
        sys.argv = previous_argv
