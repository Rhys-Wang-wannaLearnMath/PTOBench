"""Case-local adapter for the byte-exact EP8 packed prefill Fixture."""
import types
import sys

import common
import golden


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case['config'], case['case_id'].replace('.', '_') + '_fixture_config')
    alias = types.ModuleType('config')
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    official_root = (sealed / 'official').resolve()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0] if original_argv else 'fixture_builder',
                    '--ep', str(config.EP_WORLD_SIZE)]
        specs = common.load_official_fixture_specs(
            official_root / 'original.py',
            case['case_id'].replace('.', '_') + '_official',
            seed,
            config_alias=alias,
            builder_kwargs={
                'layer_id': config.LAYER_ID,
                'chunk_lens': config.CHUNK_LENS,
                'start_positions': config.START_POSITIONS,
            },
        )
    finally:
        sys.argv = original_argv

    previous_config = sys.modules.get('config')
    previous_path = sys.path[:]
    try:
        with common._isolated_module_scope((official_root,)):
            sys.modules['config'] = alias
            sys.path.insert(0, str(official_root))
            sys.argv = [original_argv[0] if original_argv else 'fixture_builder',
                        '--ep', str(config.EP_WORLD_SIZE)]
            official = common.load_module(
                official_root / 'original.py',
                case['case_id'].replace('.', '_') + '_abi',
            )
            expected_names = [*official.HOST_TENSOR_ORDER, *case['scalars']]
    finally:
        sys.argv = original_argv
        sys.path[:] = previous_path
        if previous_config is None:
            sys.modules.pop('config', None)
        else:
            sys.modules['config'] = previous_config
    if [spec.name for spec in specs] != expected_names:
        raise ValueError('official Fixture ABI differs from fixed Case')
    for spec in specs:
        if isinstance(spec, golden.TensorSpec):
            spec.direction = ('inout' if spec.name in case['state']['initial_state']
                              else 'out' if spec.name in case['correctness']['outputs'] else 'in')
    return specs
