"""Case-local FLASH adapter for byte-exact official SWA Fixture specs."""
import hashlib
from pathlib import Path
import common


# Case-bound interface. Private maintenance artifacts are not dependencies.
import types

_CASE_ID = 'deepseek_v4_flash_dspark.decode_swa_flash'
_PRESET = 'FLASH'
_OFFICIAL_SHA256 = {'original.py': 'ccab795a3f021d0e2a65052b71aeae7709f768ed28ce0131a88c5136463288e9', 'original_config.py': 'e823df2d88d5f128b928e476f94ee46eb6f3fd1c2c21c506f9334f66c74f0bd9', 'hc_pre.py': 'b5e347ed55eae5e7678c40bce3b9212c1b715419589a09b7edbcdf2a922755ee', 'hc_post.py': 'b5a57578e26b02f38c502de699ca1e9d8f86e3a8bbe43585c2f2ee9d727a48a5', 'qkv_proj_rope.py': 'cc6679d9766a8770f3f88d26da338bcaee26b25492d909507efc3d09f51f6921', 'rmsnorm.py': '776c53c970f22981e066733e7b857a71c7df42990201db5eb941f099b0e47ac0', 'decode_sparse_attn_swa.py': '518f402c7ab1195e81756a4b2c7e6b7535d6ed2e6abf54298b7c1cca2147d2ac', 'utils.py': '9b110b3d2de05e84b61d53cdcd23d658afcc02dbb769060e5cfb01b095a23cc1'}


def build(case_dir, sealed, case, seed):
    case_dir, sealed = Path(case_dir), Path(sealed)
    if case.get("case_id") != _CASE_ID:
        raise ValueError("Fixture builder Case identity mismatch")
    identity = _CASE_ID.replace(".", "_")
    config = common.load_module(case_dir / case["config"], "fixture_" + identity + "_public_config")
    if config.CASE_ID != _CASE_ID or config.CASE_CONFIG.name != _PRESET.lower():
        raise ValueError("Public config Case/preset identity mismatch")
    policy = common.read_json(sealed / "fixture_policy.json")
    if policy != {"schema_version": 1, "fixture_seeds": {"fixture_000": 0}} or seed != policy["fixture_seeds"]["fixture_000"]:
        raise ValueError("fixture_000 requires its fixed policy seed")
    official = sealed / "official"
    for name, digest in _OFFICIAL_SHA256.items():
        if hashlib.sha256((official / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Unrecognized historical source: " + name)
    original_config = common.load_module(official / "original_config.py", "fixture_" + identity + "_original_config")
    alias = types.ModuleType("fixture_" + identity + "_config_alias")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    # Import-only legacy helper bindings, excluded from the public SWA contract.
    for name in ("PREFILL_BATCH", "PREFILL_SEQ", "C4A_COMPRESSOR_BLOCK_SIZE", "C128_COMPRESSOR_BLOCK_SIZE", "KV_CMP_MAX_BLOCKS", "DECODE_CMP_BLOCK_NUM"):
        setattr(alias, name, getattr(original_config, name))
    return common.load_official_fixture_specs(official / "original.py", "fixture_" + identity + "_official", seed, config_alias=alias)
