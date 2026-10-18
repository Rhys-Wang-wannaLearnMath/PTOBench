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

MODE = 'hca'
BINDINGS = {'decode_hca': {'before_sha256': 'f7c8c2d2f2b7291585891ce175b3065e1748e0e2b092cb9f283c0928a274b136', 'after_sha256': 'e084711e12f0ae4108f0394b520020cc9c8847cae5f606386cb520bc832fe858', 'replacements': [(b'MAX_SEQ_LEN = M.max_position_embeddings\n', b'MAX_SEQ_LEN = 1_048_576\n'), (b'build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)', b'build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16, max_seq_len=MAX_SEQ_LEN)')]}, 'decode_compressor_ratio128': {'before_sha256': '48de3db04a96cec3052825abdb7509e8c487a5a0fb6971e7256f4fc60c9a79fe', 'after_sha256': '54f866e4dac6a18555dcd0363a04babe416e707c5112c9e18de60aa4f909f3ec', 'replacements': [(b'MAX_SEQ_LEN = M.max_position_embeddings\n', b'MAX_SEQ_LEN = 1_048_576\n')]}, 'decode_sparse_attn_hca': {'before_sha256': 'bd6f025f12b2ff39f861e6b7421b4d4b2cd35101615dad985d2c64a342810462', 'after_sha256': '6f2b70b601fad62e64e9c70b5fcc75c8b318cb2264f8c2ef3fce9ab70b9e59d0', 'replacements': [(b'MAX_SEQ_LEN = M.max_position_embeddings\n', b'MAX_SEQ_LEN = 1_048_576\n'), (b'VALID_TOKEN_TILE = 8\n', b'VALID_TOKEN_TILE = 4\n')]}}


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


# Canonical Case binding for the reviewed, sealed-only import contexts above.
import types


def build(case_dir, sealed, case, seed):
    """Return official deterministic specs; never generate serialized fixtures."""
    case_dir, sealed = Path(case_dir), Path(sealed)
    case_id = case["case_id"]
    identity = case_id.replace(".", "_")
    config = common.load_isolated_module(
        case_dir / case["config"], f"fixture_{identity}_public_config",
    )
    preset = config.CASE_CONFIG.name.upper()
    if config.CASE_ID != case_id or not case_id.endswith("_" + preset.lower()):
        raise ValueError("Case identity and selected public preset disagree")
    policy = common.read_json(sealed / "fixture_policy.json")
    if seed != policy["fixture_seeds"]["fixture_000"]:
        raise ValueError("Seed disagrees with canonical fixture_000 policy")
    original_config = common.load_isolated_module(
        sealed / "official" / "original_config.py",
        f"fixture_{identity}_original_config",
    )
    alias = types.ModuleType(f"fixture_{identity}_config_alias")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    # These are import/assert prerequisites for unused historical branches only.
    for name in ("PREFILL_BATCH", "PREFILL_SEQ", "C4A_COMPRESSOR_BLOCK_SIZE"):
        setattr(alias, name, getattr(original_config, name))
    return load_scoped_specs(
        sealed / "official", "original.py", f"fixture_{identity}_official", seed,
        preset=preset, config_alias=alias,
    )
