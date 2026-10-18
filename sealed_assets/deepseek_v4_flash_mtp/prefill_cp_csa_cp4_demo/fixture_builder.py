"""Official Fixture adapter for deepseek_v4_flash_mtp.prefill_cp_csa_cp4_demo.

Load the byte-exact selected source with its selected public contract.
Sampling and golden functions are unchanged.
"""

import sys
from types import ModuleType

import torch

import common


def build(case_dir, sealed, case, seed):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_fixture_config')
    root = sealed / 'official'
    historical = common.load_module(root / 'original_config.py',
                                    identity + '_original_config')
    # Historical fields serve only unused official annotations. Public bindings win.
    alias = ModuleType('config')
    alias.__dict__.update(vars(historical))
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    finder = None
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(root / 'original.py'), '--cp', str(config.CP_SIZE)]
        if finder is not None:
            sys.meta_path.insert(0, finder)
        with torch.random.fork_rng(devices=[]):
            return common.load_official_fixture_specs(
                root / 'original.py', identity + '_official', seed,
                config_alias=alias, builder_kwargs={'cp_size': config.CP_SIZE})
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)
        sys.argv[:] = previous_argv
