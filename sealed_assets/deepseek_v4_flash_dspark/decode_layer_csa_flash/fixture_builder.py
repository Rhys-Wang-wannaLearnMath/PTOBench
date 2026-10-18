# Canonical fixture adapter for deepseek_v4_flash_dspark.decode_layer_csa_flash.
"""SHA-bound Fixture/CPU import bindings for the exact recovered decode-layer closure. Never execute runtime through this adapter."""
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

IMPORT_BINDINGS = {'decode_indexer': ('2089569dfd30b72a3304bbd1bd23e25a20c855f4d326db170e228a192abfe7bc', b'WEIGHTS_OK = 4\n', b'WEIGHTS_OK = 2\n'), 'decode_o_proj': ('ef7fe2917e6707e77a15d37da6bbc1b968545f58ba1498ec19cdbfa8f0a47f5b', b'O_RS_D_TILE = 4096\n', b'O_RS_D_TILE = 1024\n'), 'gate': ('03837f608463ac3b4ae2d91ef34c7485099e026b352b3693a01ba417fa66dabf', b'GATE_D_TILE = 2048\n', b'GATE_D_TILE = 1024\n')}


@contextmanager
def _fixture_import(official_root, bindings):
    adapted = {}
    for name, (expected, old, new) in bindings.items():
        helper = Path(official_root).resolve() / (name + ".py")
        data = helper.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Unrecognized historical decode-layer helper: " + name)
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


DEMO_IMPORT_BINDINGS = {'gate': ('03837f608463ac3b4ae2d91ef34c7485099e026b352b3693a01ba417fa66dabf', b'GATE_N_TILE = 16', b'GATE_N_TILE = 2'), 'decode_hca': ('81d55ab88e3308a5579380720dfb330485a847e406c48fe3f051d7180df48cbe', b'    low = math.floor(\n        dim * math.log(original_seq_len / (int(M.beta_fast) * 2 * math.pi))\n        / (2 * math.log(base))\n    )\n    high = math.ceil(\n        dim * math.log(original_seq_len / (int(M.beta_slow) * 2 * math.pi))\n        / (2 * math.log(base))\n    )\n    low = max(low, 0)\n    high = min(high, dim - 1)\n    if low == high:\n        high += 0.001\n    ramp = torch.clamp(\n        (torch.arange(half_dim, dtype=torch.float32) - low) / (high - low), 0, 1\n)\n    smooth = 1 - ramp\n    inv_freq = inv_freq / float(M.rope_factor) * (1 - smooth) + inv_freq * smooth\n', b'    if original_seq_len > 0:\n        low = math.floor(\n            dim * math.log(original_seq_len / (int(M.beta_fast) * 2 * math.pi))\n            / (2 * math.log(base))\n        )\n        high = math.ceil(\n            dim * math.log(original_seq_len / (int(M.beta_slow) * 2 * math.pi))\n            / (2 * math.log(base))\n        )\n        low = max(low, 0)\n        high = min(high, dim - 1)\n        if low == high:\n            high += 0.001\n        ramp = torch.clamp(\n            (torch.arange(half_dim, dtype=torch.float32) - low) / (high - low), 0, 1\n    )\n        smooth = 1 - ramp\n        inv_freq = inv_freq / float(M.rope_factor) * (1 - smooth) + inv_freq * smooth\n')}

def demo_fixture_import(official_root):
    return _fixture_import(official_root, DEMO_IMPORT_BINDINGS)


# Reviewed sealed Fixture import contexts; never used for NPU execution.
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


def _config_alias(case_dir, sealed, case):
    identity = case["case_id"].replace(".", "_")
    fixed = common.load_module(case_dir / case["config"], "config_" + identity)
    # The complete byte-exact closure imports annotations for unused prefill
    # entries and unselected attention modes. Keep those source-only defaults
    # inside this temporary alias; all evaluated decode bindings come from fixed.
    alias = common.load_module(sealed / "official" / "original_config.py",
                               "fixture_config_" + identity)
    alias.__dict__.update({key: value for key, value in vars(fixed).items()
                           if not key.startswith("__")})
    for sibling in ("DEMO", "FLASH", "PRO", "PRESETS"):
        alias.__dict__.pop(sibling, None)
    alias.FLASH = fixed.CASE_CONFIG
    # Preserve raw preset experts here: official moe applies its native EP
    # reduction once. Rebind the source's preset-dependent paging expressions.
    alias.KV_ORI_MAX_BLOCKS = (fixed.CASE_CONFIG.max_position_embeddings + alias.BLOCK_SIZE - 1) // alias.BLOCK_SIZE
    alias.KV_CMP_MAX_BLOCKS = (fixed.CASE_CONFIG.max_position_embeddings // 4 + alias.BLOCK_SIZE - 1) // alias.BLOCK_SIZE
    alias.IDX_CACHE_MAX_BLOCKS = 2 * alias.KV_CMP_MAX_BLOCKS
    return fixed, alias


def build(case_dir, sealed, case, seed):
    """Return selected official specs without serializing a Fixture or running NPU."""
    case_dir, sealed = Path(case_dir), Path(sealed)
    fixed, alias = _config_alias(case_dir, sealed, case)
    if fixed.CASE_ID != case["case_id"]:
        raise ValueError("Fixture/public Case identity mismatch")
    previous_argv = sys.argv
    # Historical imports parse CLI topology. Bind it solely to public config.
    sys.argv = [previous_argv[0], "--tp=" + str(fixed.TP_SIZE),
                "--ep=" + str(fixed.EP_SIZE)]
    try:
        return load_scoped_specs(
            sealed / "official", "original.py",
            "official_" + case["case_id"].replace(".", "_"), seed,
            preset=fixed.CASE_CONFIG.name.upper(), config_alias=alias,
            builder_name=fixed.OFFICIAL_BUILDER,
            builder_kwargs={"start_pos": fixed.FIXTURE_START_POS,
                            "layer_id": fixed.LAYER_ID},
        )
    finally:
        sys.argv = previous_argv
