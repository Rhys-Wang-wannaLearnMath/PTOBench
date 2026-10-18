"""Official Fixture adapter for deepseek_v4_flash_mtp.prefill_swa_pro.

Scoped historical PRO import for Fixture support, never an NPU Ref.

The preserved hc_pre module asserts DSL-only split divisibility at import.
RMS_OK does not occur in its builder or golden functions. The corresponding
runtime repair must separately use eight complete 3584-column splits.
"""
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

SELECTED_HC_PRE_SHA256 = "39f365df3925fc555ce6f21262ac2fa528ca089b1d3264e18feaf46342c9feda"


@contextmanager
def pro_fixture_import(official_root):
    helper = Path(official_root).resolve() / "hc_pre.py"
    source = helper.read_bytes()
    if hashlib.sha256(source).hexdigest() != SELECTED_HC_PRE_SHA256:
        raise ValueError("Unrecognized historical HC helper; refuse import adaptation")
    old = b"RMS_OK = 16\n"
    if source.count(old) != 1:
        raise ValueError("Historical RMS split binding is not unique")
    adapted = source.replace(old, b"RMS_OK = 8\n")

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            exec(compile(adapted, str(helper), "exec"), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "hc_pre":
                return importlib.util.spec_from_file_location(fullname, helper, loader=Loader())
            return None

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


# Reviewed maintenance context embedded in this Case Fixture adapter.
from contextlib import nullcontext
import common


def load_scoped_specs(official_root, entry_name, module_name, seed, *, preset,
                      config_alias, builder_name="build_tensor_specs",
                      builder_args=(), builder_kwargs=None):
    factory = globals().get(preset.lower() + "_fixture_import")
    scope = factory(official_root) if factory else nullcontext()
    with scope:
        return common.load_official_fixture_specs(
            Path(official_root) / entry_name, module_name, seed,
            config_alias=config_alias, builder_name=builder_name,
            builder_args=builder_args, builder_kwargs=builder_kwargs,
        )


from types import ModuleType


def build(case_dir, sealed, case, seed):
    """Return official specs; canonical Fixture materialization belongs to Stage 3."""
    identity = case["case_id"].replace(".", "_")
    fixed = common.load_module(
        Path(case_dir) / case["config"], f"fixed_config_{identity}"
    )
    official_root = Path(sealed) / "official"
    original_config = common.load_module(
        official_root / "original_config.py", f"original_config_{identity}"
    )
    # Original deployment constants were evaluated before FLASH was rebound.
    # These five exports satisfy dormant historical imports/guards only.
    alias = ModuleType(f"fixture_config_{identity}")
    for name in (
        "DECODE_BATCH", "DECODE_SEQ", "DECODE_START_POS",
        "C4A_COMPRESSOR_BLOCK_SIZE", "C128_COMPRESSOR_BLOCK_SIZE",
    ):
        setattr(alias, name, getattr(original_config, name))
    alias.FLASH = fixed.CASE_CONFIG
    for name in (
        "BLOCK_SIZE", "INT8_AMAX_EPS", "INT8_SCALE_MAX", "FP32_NEG_INF",
        "PREFILL_BATCH", "PREFILL_SEQ", "PREFILL_TOKENS",
        "KV_ORI_TABLE_MAX_BLOCKS", "KV_ORI_MAX_BLOCKS", "KV_CMP_MAX_BLOCKS",
        "ORI_KV_BLOCK_NUM", "CMP_KV_BLOCK_NUM", "DECODE_ORI_BLOCK_NUM",
        "DECODE_CMP_BLOCK_NUM", "PREFILL_ORI_MAX_BLOCKS", "PREFILL_ORI_BLOCK_NUM",
        "PREFILL_CMP_MAX_BLOCKS", "PREFILL_CMP_BLOCK_NUM",
    ):
        setattr(alias, name, getattr(fixed, name))
    return load_scoped_specs(
        official_root, "original.py", f"official_{identity}", seed,
        preset=fixed.CASE_CONFIG.name.upper(), config_alias=alias,
        builder_kwargs={"start_pos": fixed.START_POS, "num_tokens": fixed.NUM_TOKENS},
    )
