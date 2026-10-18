"""Official Fixture adapter for the PRO packed CSA prefill contract."""

import sys

import common


def build(case_dir, sealed, case, seed):
    config = common.load_module(case_dir / case["config"], "prefill_attention_csa_pro_fixture_config")
    alias = type(sys)("config")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    official_root = str(sealed / "official")
    local_names = {p.stem for p in (sealed / "official").glob("*.py")}
    saved = {n: sys.modules[n] for n in local_names if n in sys.modules}
    for n in saved:
        del sys.modules[n]
    sys.path.insert(0, official_root)
    try:
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py",
            "prefill_attention_csa_pro_official",
            seed,
            config_alias=alias,
        )
    finally:
        sys.path.remove(official_root)
        for n in local_names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
