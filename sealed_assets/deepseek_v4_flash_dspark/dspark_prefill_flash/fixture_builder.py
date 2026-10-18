"""Official Fixture adapter for deepseek_v4_flash_dspark.dspark_prefill_flash."""
from pathlib import Path
import sys
from types import ModuleType

import common


def build(case_dir, sealed, case, seed):
    """Bind the official builder without materializing tensors or running a kernel."""
    case_dir, sealed = Path(case_dir), Path(sealed)
    identity = case["case_id"].replace(".", "_")
    fixed = common.load_isolated_module(
        case_dir / case["config"], identity + "__fixture_public_config",
    )
    if fixed.CASE_ID != case["case_id"]:
        raise ValueError("Fixture config identity does not match case.json")
    historical = common.load_isolated_module(
        sealed / "official" / "original_config.py",
        identity + "__fixture_original_config",
    )
    alias = ModuleType("config")
    alias.__dict__.update({key: value for key, value in vars(fixed).items()
                           if not key.startswith("__")})
    # Byte-exact moe.py rescales this original selected object exactly once.
    alias.FLASH = fixed.CASE_CONFIG
    # Standalone import defaults never replace this Case's public batch/sequence.
    for key in ("PREFILL_BATCH", "PREFILL_TOKENS"):
        setattr(alias, key, getattr(historical, key))
    previous_argv = sys.argv
    try:
        sys.argv = ["official_fixture", "--ep", str(fixed.EP)]
        return common.load_official_fixture_specs(
            sealed / "official" / "original.py", identity + "__fixture_official", seed,
            config_alias=alias, builder_args=(fixed.FIXTURE_BATCH,),
            builder_kwargs={"mode": fixed.FIXTURE_MODE},
        )
    finally:
        sys.argv = previous_argv
