"""Official Fixture adapter for deepseek_v4_flash_dspark.dspark_drafter_pro. Specs only; no NPU execution."""
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

IMPORT_BINDINGS = {
    "hc_pre": ("76d202d353c23ffbca986064e304253693093ba09bd90233805aa9937dc32dc0",
               b"RMS_OK = 16\n", b"RMS_OK = 8\n"),
    "gate": ("407df9874c50bf19885d279bd92ba0d2368b91e2a97468571193f297868ad228",
             b"GATE_D_TILE = 2048\n", b"GATE_D_TILE = 1024\n"),
}


@contextmanager
def pro_fixture_import(official_root):
    adapted = {}
    for name, (expected, old, new) in IMPORT_BINDINGS.items():
        helper = Path(official_root).resolve() / (name + ".py")
        data = helper.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Unrecognized historical drafter helper: " + name)
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


# The reviewed import scope is embedded in this sealed Fixture adapter.
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

# Only preserved source import-time compatibility uses these unrelated fields.
_LEGACY_IMPORT_ONLY = (
    "PREFILL_BATCH", "PREFILL_SEQ", "PREFILL_TOKENS",
    "C4A_COMPRESSOR_BLOCK_SIZE", "C128_COMPRESSOR_BLOCK_SIZE", "DECODE_START_POS",
)


def build(case_dir, sealed, case, seed):
    """Return official specs bound to the canonical preset/batch/EP; no execution."""
    case_dir, sealed = Path(case_dir), Path(sealed)
    identity = case["case_id"].replace(".", "_")
    fixed = common.load_isolated_module(
        case_dir / case["config"], "fixture_config_" + identity,
    )
    if fixed.CASE_ID != case["case_id"]:
        raise ValueError("Fixture config identity does not match case.json")
    policy = common.read_json(sealed / "fixture_policy.json")
    if seed != fixed.FIXTURE_SEED or seed != policy["fixture_seeds"]["fixture_000"]:
        raise ValueError("Fixture seed does not match the fixed contract")
    official_root = sealed / "official"
    provenance = common.read_json(official_root / "source_provenance.json")
    original_config = next(row for row in provenance["dependencies"]
                           if row["preserved_path"] == "official/original_config.py")
    for row in (provenance["source"], original_config):
        if common.sha256_file(sealed / row["preserved_path"]) != row["preserved_sha256"]:
            raise ValueError("Preserved official source hash does not match provenance")
    historical = common.load_isolated_module(
        official_root / "original_config.py", "fixture_original_config_" + identity,
    )
    alias = ModuleType("config")
    alias.__dict__.update({key: value for key, value in vars(fixed).items()
                           if not key.startswith("__")})
    alias.FLASH = fixed.CASE_CONFIG
    for key in _LEGACY_IMPORT_ONLY:
        setattr(alias, key, getattr(historical, key))
    previous_argv = sys.argv
    try:
        sys.argv = ["official_fixture", "--ep", str(fixed.EP)]
        return load_scoped_specs(
            official_root, "original.py", "official_" + identity, seed,
            preset=fixed.CASE_CONFIG.name, config_alias=alias,
            builder_name="build_tensor_specs", builder_args=(fixed.BATCH,),
        )
    finally:
        sys.argv = previous_argv
