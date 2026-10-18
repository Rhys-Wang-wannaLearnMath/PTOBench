"""Reviewed PRO Fixture/CPU import support; never an NPU implementation."""
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

SELECTED_HC_PRE_SHA256 = "b5e347ed55eae5e7678c40bce3b9212c1b715419589a09b7edbcdf2a922755ee"


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


"""Reviewed DEMO Fixture/CPU import adaptation. Never an NPU runtime loader."""
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

MODE = 'swa'
BINDINGS = {'decode_swa': {'before_sha256': 'ccab795a3f021d0e2a65052b71aeae7709f768ed28ce0131a88c5136463288e9', 'after_sha256': '59e117e0d9bc084b62f0ddf44c479ce70f8933690ccf93e08383f86345a50629', 'replacements': [(b'MAX_SEQ_LEN = M.max_position_embeddings\n', b'MAX_SEQ_LEN = 1_048_576\n'), (b'build_rope_tables(M, 0, dtype=torch.bfloat16)', b'build_rope_tables(M, 0, dtype=torch.bfloat16, max_seq_len=MAX_SEQ_LEN)')]}, 'decode_sparse_attn_swa': {'before_sha256': '518f402c7ab1195e81756a4b2c7e6b7535d6ed2e6abf54298b7c1cca2147d2ac', 'after_sha256': '457ec7a110f6718e873414c70d239e3a7e4b679b83ac69bf6c3860a6b2d22a7e', 'replacements': [(b'MAX_SEQ_LEN = M.max_position_embeddings\n', b'MAX_SEQ_LEN = 1_048_576\n')]}}


@contextmanager
def demo_fixture_import(official_root):
    import common

    root = Path(official_root).resolve()
    entry = root / ("original.py" if (root / "original.py").exists() else "decode_" + MODE + ".py")
    adapted = {}
    for name, binding in BINDINGS.items():
        path = entry if name == "decode_" + MODE else root / (name + ".py")
        original = path.read_bytes()
        if hashlib.sha256(original).hexdigest() != binding["before_sha256"]:
            raise ValueError("Unrecognized historical DEMO source: " + name)
        data = original
        for old, new in binding["replacements"]:
            if data.count(old) != 1:
                raise ValueError("Nonunique historical DEMO adaptation: " + name)
            data = data.replace(old, new)
        if hashlib.sha256(data).hexdigest() != binding["after_sha256"]:
            raise ValueError("Unrecognized DEMO adapted bytes: " + name)
        adapted[name] = (path, data)

    class Loader(importlib.abc.Loader):
        def __init__(self, path, data):
            self.path, self.data = path, data

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            exec(compile(self.data, str(self.path), "exec"), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in adapted:
                helper, data = adapted[fullname]
                return importlib.util.spec_from_file_location(fullname, helper, loader=Loader(helper, data))
            return None

    original_load = common.load_module
    def load_reviewed_entry(path, name, search_paths=()):
        if Path(path).resolve() != entry:
            return original_load(path, name, search_paths)
        _, data = adapted["decode_" + MODE]
        spec = importlib.util.spec_from_file_location(name, entry, loader=Loader(entry, data))
        module = importlib.util.module_from_spec(spec)
        added = [*(str(value) for value in search_paths), str(entry.parent)]
        previous_bytecode = sys.dont_write_bytecode
        sys.path[:0] = added
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_bytecode
            del sys.path[:len(added)]
        return module

    finder = Finder()
    with common._isolated_module_scope((root,)):
        sys.meta_path.insert(0, finder)
        common.load_module = load_reviewed_entry
        try:
            yield
        finally:
            common.load_module = original_load
            sys.meta_path.remove(finder)


# Fixture-only placement proposal; Stage2 supplies its normal build interface.
from contextlib import nullcontext
import common


def load_scoped_specs(official_root, entry_name, module_name, seed, *, preset,
                      config_alias):
    if preset not in {"DEMO", "FLASH", "PRO"}:
        raise ValueError("Unsupported preset")
    factory = globals().get(preset.lower() + "_fixture_import")
    with factory(official_root) if factory else nullcontext():
        return common.load_official_fixture_specs(
            Path(official_root) / entry_name, module_name, seed,
            config_alias=config_alias,
        )


# Case-bound interface. Private maintenance artifacts are not dependencies.
import types

_CASE_ID = 'deepseek_v4_flash_dspark.decode_swa_pro'
_PRESET = 'PRO'
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
    return load_scoped_specs(official, "original.py", "fixture_" + identity + "_official", seed, preset=_PRESET, config_alias=alias)
