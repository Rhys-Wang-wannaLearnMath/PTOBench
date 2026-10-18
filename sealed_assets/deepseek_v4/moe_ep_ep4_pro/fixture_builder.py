"""Case-local adapter for the byte-exact official Fixture builder."""
import ast
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import common

class _GateLoader(importlib.abc.Loader):
    def __init__(self, path): self.path = path
    def create_module(self, spec): return None
    def exec_module(self, module):
        tree = ast.parse(self.path.read_text(), filename=str(self.path))
        removed = [n for n in tree.body if isinstance(n, ast.Assert) and ast.unparse(n.test) == 'N_EXPERTS % GATE_N_TILE == 0']
        assert len(removed) == 1
        tree.body = [n for n in tree.body if n not in removed]
        module.__file__ = str(self.path)
        module.__manual_removed_assertions__ = ['N_EXPERTS % GATE_N_TILE == 0']
        exec(compile(tree, str(self.path), 'exec'), module.__dict__)

class _GateFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path): self.path = path
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'gate':
            return importlib.util.spec_from_file_location(fullname, self.path, loader=_GateLoader(self.path))
        return None

def build(case_dir, sealed, case, seed):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_fixture_config')
    alias = ModuleType('config')
    alias.__dict__.update(vars(config))
    preset = case['case_id'].rsplit('_', 1)[-1].upper()
    # original.py historically imports FLASH and performs the single EP scaling.
    alias.FLASH = getattr(config, preset)
    official_root = Path(sealed) / 'official'
    finder = _GateFinder(official_root / 'gate.py') if config.N_EXPERTS_GLOBAL < 16 else None
    # The official source freezes rank-dependent shapes during import.
    # Bind only this Case's fixed CLI input for the complete helper lifetime;
    # caller arguments (including a conflicting --ep) must not change the ABI.
    previous_argv = sys.argv
    try:
        sys.argv = [str(official_root / 'original.py'), '--ep', str(config.EP_WORLD_SIZE)]
        if finder is not None: sys.meta_path.insert(0, finder)
        return common.load_official_fixture_specs(
            official_root / 'original.py', identity + '_official', seed, config_alias=alias,
            builder_kwargs={'layer_id': config.LAYER_ID},
        )
    finally:
        sys.argv = previous_argv
        if finder is not None: sys.meta_path.remove(finder)
