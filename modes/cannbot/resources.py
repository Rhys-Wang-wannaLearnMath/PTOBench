"""Frozen upstream bytes and original role/skill locations."""
import hashlib
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROLES = {
    'pypto-op-orchestrator': 'plugins-official/pypto-op-orchestrator/AGENTS.md',
    'pypto-op-optimizer': 'plugins-official/pypto-op-orchestrator/agents/pypto-op-optimizer.md',
    'pypto-op-verifier': 'plugins-official/pypto-op-orchestrator/agents/pypto-op-verifier.md',
}


def manifest():
    return json.loads((DIRECTORY / 'upstream.lock.json').read_text())


def freeze(destination):
    """Copy original resources; never inspect API or profiling compatibility."""
    lock = manifest()
    for name, expected in lock['files'].items():
        source = DIRECTORY / 'upstream' / name
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('frozen CANNBot resource bytes changed')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(source.stat().st_mode & 0o777 & ~0o222)
    return lock


def skills(upstream):
    """Index immutable skills and their progressive reference documents."""
    result = {}
    for source in sorted((upstream / 'ops').glob('**/SKILL.md')):
        name = source.parent.name
        for line in source.read_text().splitlines():
            if line.startswith('name:'):
                name = line.split(':', 1)[1].strip().strip('\"\'')
                break
        if name in result:
            raise ValueError('duplicate upstream skill identity')
        result[name] = str(source)
        # Agents request progressive-disclosure documents as
        # ``<skill-name>/<relative-path>``. Keep the index rooted in the
        # frozen upstream tree so this remains a read-only resource lookup.
        for reference in sorted(source.parent.rglob('*')):
            if reference.is_file() and reference != source:
                relative = reference.relative_to(source.parent).as_posix()
                key = f'{name}/{relative}'
                if key in result:
                    raise ValueError('duplicate upstream skill reference identity')
                result[key] = str(reference)
    return result
