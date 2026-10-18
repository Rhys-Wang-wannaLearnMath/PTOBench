"""Official Fixture adapter for deepseek_v4_flash_dspark.prefill_layer_csa_demo; no runtime execution."""
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

import common

IMPORT_BINDINGS = {'gate': ('03837f608463ac3b4ae2d91ef34c7485099e026b352b3693a01ba417fa66dabf', b'GATE_N_TILE = 16', b'GATE_N_TILE = 2'), 'prefill_fwd': ('d99e7926417395bce77e8a0558fa03a61c2c9c755758ba604d5f0494c4bc35f6', b'if MODEL_NUM_LAYERS != FWD_NUM_LAYERS:\n    raise ValueError("DeepSeek-V4 Flash hidden layer count changed")\n', b'# Single-layer builder import only; full-forward43-layer execution forbidden.\n'), 'prefill_indexer': ('8c1ec96373d9f589e2e82b6433dc3d63af031cd1cac2a79ca5b1c40c11391cee', b'assert RAW_TOKEN_CAP <= MAX_SEQ_LEN\n', b'assert RAW_TOKEN_CAP >= min(MAX_SEQ_LEN, PREFILL_MAX_TOKENS)\n')}

@contextmanager
def _fixture_import(official_root, bindings):
    adapted = {}
    for name, (expected, old, new) in bindings.items():
        helper = Path(official_root).resolve() / (name + ".py")
        data = helper.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Unrecognized historical prefill-layer helper: " + name)
        if data.count(old) != 1:
            raise ValueError("Historical tiling binding is not unique: " + name)
        adapted[name] = (helper, data.replace(old, new))

    class Loader(importlib.abc.Loader):
        def __init__(self, helper, data):
            self.helper, self.data = helper, data

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            exec(compile(self.data, str(self.helper), "exec"), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in adapted:
                helper, data = adapted[fullname]
                return importlib.util.spec_from_file_location(fullname, helper, loader=Loader(helper, data))
            return None

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def build(case_dir, sealed, case, seed):
    """Return official specs; fixed inputs come from this Case's public config."""
    identity = case["case_id"].replace(".", "_")
    public = common.load_isolated_module(
        Path(case_dir) / case["config"], identity + "_fixture_public_config",
    )
    if case["case_id"] != public.CASE_ID:
        raise ValueError("Fixture Case identity differs from the public contract")
    official_root = Path(sealed) / "official"
    config_alias = common.load_isolated_module(
        official_root / "original_config.py", identity + "_fixture_historical_config",
    )
    for name, value in vars(public).items():
        if not name.startswith("__"):
            setattr(config_alias, name, value)
    # Historical moe.py performs the expert reduction once after this raw binding.
    config_alias.FLASH = public.CASE_CONFIG
    config_alias.EP = public.SOURCE_EP
    config_alias.MOE_TOKENS = public.MOE_TOKENS
    config_alias.KV_ORI_MAX_BLOCKS = public.KV_ORI_MAX_BLOCKS
    config_alias.KV_CMP_MAX_BLOCKS = public.KV_CMP_MAX_BLOCKS
    config_alias.IDX_CACHE_MAX_BLOCKS = public.IDX_CACHE_MAX_BLOCKS
    previous_argv = sys.argv
    sys.argv = [identity + "_fixture", "--ep", str(public.EP), "--tp", str(public.TP_SIZE)]
    try:
        scope = _fixture_import(official_root, IMPORT_BINDINGS)
        with scope:
            return common.load_official_fixture_specs(
                official_root / "original.py", identity + "_fixture_official", seed,
                config_alias=config_alias, builder_name="build_tensor_specs",
                builder_args=(), builder_kwargs={"layer_id": public.LAYER_ID},
            )
    finally:
        sys.argv = previous_argv
