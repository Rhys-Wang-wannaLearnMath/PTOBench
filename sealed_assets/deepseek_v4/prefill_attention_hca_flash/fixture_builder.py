'Build prefill_attention_hca_flash fixtures through its byte-exact selected official closure.'
import sys
import common

def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "deepseek_v4_prefill_attention_hca_flash_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    official_root = str(sealed / "official")
    local_names = {p.stem for p in (sealed / "official").glob("*.py")}
    saved = {name: sys.modules[name] for name in local_names if name in sys.modules}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, official_root)
    try:
        return common.load_official_fixture_specs(sealed / "official" / "original.py", "deepseek_v4_prefill_attention_hca_flash_official", seed, config_alias=alias)
    finally:
        sys.path.remove(official_root)
        for name in local_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
