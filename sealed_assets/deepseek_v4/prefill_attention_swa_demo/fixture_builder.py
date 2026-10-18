"""Official Fixture adapter for deepseek_v4.prefill_attention_swa_demo."""
import types
import common


def build(case_dir, sealed, case, seed):
    if case['case_id'] != 'deepseek_v4.prefill_attention_swa_demo':
        raise ValueError('Fixture adapter requires deepseek_v4.prefill_attention_swa_demo')
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_fixture_config')
    assert config.CASE_CONFIG is config.DEMO
    assert (config.START_POS, config.NUM_TOKENS, config.PREFILL_BATCH, config.PREFILL_SEQ) == (0, 128, 1, 128)
    alias = types.ModuleType('config')
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    return common.load_official_fixture_specs(sealed / 'official/original.py',
                                              identity + '_official_fixture', seed,
                                              config_alias=alias)
