"""Method-owned reference bundles; the evaluator has no knowledge policy."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

from modes._shared.config import ConfigError

SNAPSHOT = 'method/resources/raw_wiki'


def describe(snapshot):
    """Verify the original corpus manifest and fingerprint the complete tree."""
    snapshot = Path(snapshot)
    try:
        if not stat.S_ISDIR(snapshot.lstat().st_mode):
            raise ValueError('knowledge root must be a directory')
        files, directories = {}, []
        for parent, children, names in os.walk(snapshot, followlinks=False):
            parent = Path(parent)
            for name in children:
                path = parent / name
                if not stat.S_ISDIR(path.lstat().st_mode):
                    raise ValueError('knowledge contains a link or special node')
                directories.append(path.relative_to(snapshot).as_posix())
            for name in names:
                path = parent / name
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError('knowledge contains a link or special node')
                files[path.relative_to(snapshot).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        original = json.loads((snapshot / 'manifest.json').read_text())
        expected = {item['path']: item['sha256'] for item in original['files']}
        if (len(expected) != len(original['files']) or 'manifest.json' in expected
                or set(files) != set(expected) | {'manifest.json'}
                or any(files[name] != digest for name, digest in expected.items())):
            raise ValueError('knowledge does not match its original manifest')
        payload = json.dumps({'files': files, 'directories': sorted(directories)},
                             sort_keys=True, separators=(',', ':')).encode()
        return {'schema_version': 1, 'name': 'raw_wiki', 'source': 'pto_wiki/raw_wiki',
                'source_manifest_sha256': files['manifest.json'],
                'content_sha256': hashlib.sha256(payload).hexdigest(),
                'version': {key: original[key] for key in
                            ('edition', 'captured_on', 'pypto_revision')}}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ConfigError('raw_wiki reference bundle is unavailable or changed') from error


def validate(mode_directory):
    directory = Path(mode_directory) / 'knowledge'
    try:
        lock = json.loads((directory / 'knowledge.lock.json').read_text())
        if describe(directory / 'raw_wiki') != lock:
            raise ValueError('knowledge version differs from its pin')
        return lock
    except (OSError, ValueError, TypeError) as error:
        raise ConfigError('mode raw_wiki reference bundle does not match its pin') from error


def freeze(mode_directory, session_root):
    """Make independent files once per Bench Session, then remove write access."""
    lock = validate(mode_directory)
    source = Path(mode_directory) / 'knowledge/raw_wiki'
    destination = Path(session_root) / SNAPSHOT
    if destination.exists() or destination.is_symlink():
        raise ConfigError('session raw_wiki snapshot already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=True)
    if describe(destination) != lock:
        raise ConfigError('copied raw_wiki snapshot differs from its pinned source')
    for parent, children, files in os.walk(destination, topdown=False):
        for name in files:
            (Path(parent) / name).chmod(0o444)
        Path(parent).chmod(0o555)
    return {**lock, 'snapshot': SNAPSHOT}


def notice(session_root):
    return (f"Read {Path(session_root) / 'method/README.md'} for your read-only reference knowledge. "
            "You may read or search it as needed.\n")


def guide(session_root):
    snapshot = Path(session_root) / SNAPSHOT
    return f"""# Reference knowledge

Read-only knowledge: `{snapshot}`. Start with its `README.md`.

- `01_pypto_lib/`: PyPTO-Lib programming, execution and performance tuning.
- `02_pypto/`: PyPTO language, API, compiler and runtime documentation.
- `03_npu_910b/`: Ascend 910B architecture, synchronization and debugging.

These are fixed original documents. Read or search them with your normal tools;
reading uses no hardware budget. Keep the documents unchanged. Links outside
this tree refer to upstream resources and may not exist in this Session.
The same snapshot is available from every workspace and Agent stage in this
Bench Session. Version information is in `{Path(session_root) / 'method/knowledge.json'}`.
Use the Case contract and public Bench commands in `.benchmark_context/README.md`
for this experiment, including when a document contains other execution examples.
"""
