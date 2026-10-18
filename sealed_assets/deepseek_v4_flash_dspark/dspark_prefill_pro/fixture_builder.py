"""Official Fixture adapter for deepseek_v4_flash_dspark.dspark_prefill_pro."""
from pathlib import Path
import sys
from types import ModuleType

import common
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util

IMPORT_BINDINGS = {
    "hc_pre": ("4ddcf3fcb6769d3bd28e463c5307bf0cd6945f6ba797b4f4b23473b64a1787ff",
               b"RMS_OK = 16\n", b"RMS_OK = 8\n"),
    "gate": ("03837f608463ac3b4ae2d91ef34c7485099e026b352b3693a01ba417fa66dabf",
             b"GATE_D_TILE = 2048\n", b"GATE_D_TILE = 1024\n"),
}

@contextmanager
def _fixture_import(official_root, bindings):
    adapted = {}
    for name, (expected, old, new) in bindings.items():
        helper = Path(official_root).resolve() / (name + ".py")
        data = helper.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Unrecognized historical prefill helper: " + name)
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

def pro_fixture_import(official_root):
    return _fixture_import(official_root, IMPORT_BINDINGS)


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
        with pro_fixture_import(sealed / "official"):
            return common.load_official_fixture_specs(
                sealed / "official" / "original.py", identity + "__fixture_official", seed,
                config_alias=alias, builder_args=(fixed.FIXTURE_BATCH,),
                builder_kwargs={"mode": fixed.FIXTURE_MODE},
            )
    finally:
        sys.argv = previous_argv
