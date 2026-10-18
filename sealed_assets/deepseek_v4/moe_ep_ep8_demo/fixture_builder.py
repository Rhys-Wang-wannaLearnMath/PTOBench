"""Complete scoped official EP8 Fixture construction."""
import sys
from types import ModuleType
import common


def build(case_dir, sealed, case, seed):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_fixture_config')
    alias = ModuleType('config')
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    previous = sys.argv
    try:
        sys.argv = [str(sealed / 'official/original.py'), '--ep', str(config.EP_WORLD_SIZE)]
        return common.load_official_fixture_specs(sealed / 'official/original.py',
                                                 identity + '_official', seed, config_alias=alias)
    finally:
        sys.argv = previous
