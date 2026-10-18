"""Reviewed PRO HCA Fixture/CPU import bindings; never an NPU implementation.

Original function bodies and source bytes remain intact. These adaptations
provide full-context metadata widths for HCA (which has no indexer), RMS split
alignment, and bounded bias tiles in the owning historical module namespace.
The caller supplies the independently reviewed full-context paging config.
"""
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

import common

BINDINGS = {
    "hc_pre": (
        "b5e347ed55eae5e7678c40bce3b9212c1b715419589a09b7edbcdf2a922755ee",
        [(b"RMS_OK = 16\n", b"RMS_OK = 8\n")],
    ),
    "prefill_sparse_attn": (
        "6c4c0ab9d5d23e4681a226d341a0430b86d3107dd2b0b73f0f535163496bb24a",
        [(b"IDX_TOPK = M.index_topk\n", b"IDX_TOPK = (M.max_position_embeddings + 128 - 1) // 128\n"),
         (b"PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))\n", b"PREFILL_MAX_COMPRESSED = IDX_TOPK\n"),
         (b"BIAS_TOKEN_TILE = 16\n", b"BIAS_TOKEN_TILE = 1\n")],
    ),
    "prefill_hca": (
        "e0058e6c3e5c2a4648d52a0746aed8ca39ca9a1b5e89f27183f0875a7827d6eb",
        [(b"IDX_TOPK = M.index_topk\n", b"IDX_TOPK = (M.max_position_embeddings + 128 - 1) // 128\n"),
         (b"PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))\n", b"PREFILL_MAX_COMPRESSED = IDX_TOPK\n")],
    ),
}


def _adapt(path, binding):
    expected, replacements = binding
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Unrecognized historical HCA source: " + str(path))
    for old, new in replacements:
        if data.count(old) != 1:
            raise ValueError("Historical HCA binding is not unique: " + str(path))
        data = data.replace(old, new)
    return data


class _Loader(importlib.abc.Loader):
    def __init__(self, path, data):
        self.path, self.data = path, data

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.data, str(self.path), "exec"), module.__dict__)


@contextmanager
def adapted_entry(official_root):
    """Yield an isolated preserved-source host module with reviewed globals."""
    official_root = Path(official_root).resolve()
    entry_path = official_root / "original.py"
    if not entry_path.is_file():
        entry_path = official_root / "prefill_hca.py"
    entry_data = _adapt(entry_path, BINDINGS["prefill_hca"])
    helpers = {name: (official_root / (name + ".py"), _adapt(official_root / (name + ".py"), binding))
               for name, binding in BINDINGS.items() if name != "prefill_hca"}
    selected_config = sys.modules["config"]
    previous_path = sys.path[:]
    previous_bytecode = sys.dont_write_bytecode

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in helpers:
                source, data = helpers[fullname]
                return importlib.util.spec_from_file_location(fullname, source, loader=_Loader(source, data))
            return None

    finder = Finder()
    with common._isolated_module_scope((official_root,)):
        sys.modules["config"] = selected_config
        sys.path.insert(0, str(official_root))
        sys.meta_path.insert(0, finder)
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location("reviewed_hca_host_entry", entry_path,
                                                         loader=_Loader(entry_path, entry_data))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.meta_path.remove(finder)
            sys.path[:] = previous_path
            sys.dont_write_bytecode = previous_bytecode


def build_tensor_specs(official_root, *, start_pos=0, num_tokens=128):
    """Delegate to the original builder under the host-only import context."""
    with adapted_entry(official_root) as module:
        return module.build_tensor_specs(start_pos=start_pos, num_tokens=num_tokens)


import types

def build(case_dir, sealed, case, seed):
    case_dir, sealed = Path(case_dir), Path(sealed)
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], "fixture_" + identity + "_public_config")
    if config.CASE_ID != case["case_id"]:
        raise ValueError("Case identity mismatch")
    expected_seed = common.read_json(sealed / "fixture_policy.json")["fixture_seeds"]["fixture_000"]
    if seed != expected_seed:
        raise ValueError("Fixture seed mismatch")
    original_config = common.load_module(sealed / "official" / "original_config.py", "fixture_" + identity + "_original_config")
    alias = types.ModuleType("fixture_" + identity + "_config_alias")
    alias.__dict__.update(vars(config))
    alias.FLASH = config.CASE_CONFIG
    for key in ("DECODE_BATCH", "DECODE_SEQ", "DECODE_START_POS", "C4A_COMPRESSOR_BLOCK_SIZE"):
        setattr(alias, key, getattr(original_config, key))
    return common.load_official_fixture_specs(Path(__file__), "fixture_" + identity + "_reviewed_host", seed, config_alias=alias, builder_name="build_tensor_specs", builder_kwargs={"official_root": sealed / "official", "start_pos": config.START_POS, "num_tokens": config.NUM_TOKENS})
