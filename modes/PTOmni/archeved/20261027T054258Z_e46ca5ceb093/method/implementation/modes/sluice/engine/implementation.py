"""Freeze executable method code through the existing participant archive protocol."""

from pathlib import Path
import shutil

from ..foundation.protocol import ProtocolError
from ..foundation.resources import _resource_manifest


def freeze(source: Path, destination: Path, *, profile_tool=False) -> dict:
    if destination.exists():
        raise ProtocolError("SLUICE implementation is already prepared")
    files = [source / "runner.py", source / "runtime.py"]
    for directory, suffix in (("engine", "*.py"), ("foundation", "*.py"), ("bridge", "*.ts")):
        files.extend(sorted((source / directory).glob(suffix)))
    runtime_skills = source / "resources" / "runtime_skills"
    required_skills = (runtime_skills / "core_runtime.md", runtime_skills / "round_contract.md")
    if any(not path.is_file() or path.is_symlink() for path in required_skills):
        raise ProtocolError("Frozen SLUICE runtime contracts are incomplete")
    files.extend(required_skills)
    # Only the read-only knowledge runtime is frozen, with all of its local imports.
    # Source adapters, Agent execution, CLI and probe runners remain offline assets.
    files.extend(source / "knowledge_builder" / name for name in (
        "__init__.py", "bindings.py", "contracts.py", "environment.py", "publish.py", "retrieval.py",
        "review.py", "scheduler.py",
    ))
    if profile_tool:
        # The facade calls these SLUICE-owned helpers inside the frozen parent.
        # Shipping only profile_db leaves those public tools unable to import.
        files.extend(source / 'profiling' / name for name in (
            'execution_work.py', 'diagnostic_integrity.py', 'pmu_samples.py',
            'device_identity.py', 'device_environment.py',
        ))
    for path in files:
        if path.is_symlink():
            raise ProtocolError("Implementation code cannot be a symlink")
        target = destination / "modes/sluice" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    # Keep shared support modules on the protected public runtime path.
    (destination / "modes/__init__.py").write_text(
        '"""Frozen participant with shared PTOBench support modules."""\n'
        "from pkgutil import extend_path\n"
        "__path__ = extend_path(__path__, __name__)\n"
    )
    # Resolve SLUICE entirely inside the snapshot even if a host development
    # package later acquires an __init__.py on the shared namespace path.
    (destination / "modes/sluice/__init__.py").write_text('"""Frozen SLUICE implementation."""\n')
    if profile_tool:
        # Ship the one canonical core implementation with SLUICE. No import of
        # an external checkout is required in the protected participant.
        from .profile_core import package_root
        package = package_root() / 'profile_db'
        shutil.copytree(package, destination / 'profile_db',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc',
                                                     'render', 'cli.py', 'mcp_server.py', '__main__.py'))
    return _resource_manifest(destination)
