"""Independent Wiki snapshots and a Broker-owned readable Workspace copy."""
import grp
import hashlib
import json
import os
from pathlib import Path
import stat


def inventory(root: Path):
    root = Path(root)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError('Wiki root must be a directory, not a link or special node')
    files, directories = {}, ['']
    for parent, children, names in os.walk(root, followlinks=False):
        for name in children:
            path = Path(parent) / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError('Wiki contains a link or special node')
            directories.append(path.relative_to(root).as_posix())
        for name in names:
            path = Path(parent) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError('Wiki contains a link or special node')
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files, sorted(directories)


def readonly(root: Path):
    files, directories = inventory(root)
    for relative in files:
        (root / relative).chmod(0o440)
    for relative in reversed(directories):
        (root / relative).chmod(0o550)


def freeze_wiki(source: Path, session_root: Path, workspace: Path) -> dict:
    source, session_root, workspace = map(Path, (source, session_root, workspace))
    files, directories = inventory(source)
    if 'INDEX.md' not in files:
        raise ValueError('Wiki INDEX.md is missing')
    snapshot = session_root / 'method/resources/pypto_wiki'
    visible = workspace / '.mode_resources/pypto_wiki'
    if snapshot.exists() or snapshot.is_symlink() or visible.parent.exists() or visible.parent.is_symlink():
        raise ValueError('Wiki destinations already exist')
    gid = grp.getgrnam('ptobench-share').gr_gid
    visible.parent.mkdir(mode=0o750)
    os.chown(visible.parent, -1, gid)
    visible.parent.chmod(0o750)
    entries = []
    for target in (snapshot, visible):
        target.parent.mkdir(parents=True, exist_ok=True)
        for relative in directories:
            path = target / relative
            path.mkdir(exist_ok=True)
            os.chown(path, -1, gid)
            path.chmod(0o750)
        for relative, digest in files.items():
            data = (source / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Wiki changed while freezing')
            path = target / relative
            path.write_bytes(data)
            os.chown(path, -1, gid)
            path.chmod(0o640)
    for relative, digest in sorted(files.items()):
        entries.append({'path': relative, 'source_sha256': digest,
                        'snapshot_sha256': hashlib.sha256((snapshot / relative).read_bytes()).hexdigest(),
                        'visible_sha256': hashlib.sha256((visible / relative).read_bytes()).hexdigest(),
                        'source_path': str(Path('pto_wiki/pypto_wiki') / relative),
                        'snapshot_path': str(snapshot.relative_to(session_root) / relative),
                        'visible_path': str(visible.relative_to(workspace) / relative)})
    readonly(snapshot)
    return {'schema_version': 1, 'source': str(source), 'snapshot_path': str(snapshot),
            'visible_path': str(visible.absolute()), 'directories': directories, 'files': entries,
            'content_sha256': hashlib.sha256(json.dumps({'files': files, 'directories': directories}, sort_keys=True).encode()).hexdigest()}


def verify_wiki(manifest: dict) -> dict:
    mismatches = []
    expected = {e['path']: e['source_sha256'] for e in manifest['files']}
    for key in ('snapshot_path', 'visible_path'):
        try:
            files, directories = inventory(Path(manifest[key]))
            if files != expected or directories != manifest['directories']:
                mismatches.append(key)
        except (OSError, ValueError):
            mismatches.append(key)
    return {'unchanged': not mismatches, 'mismatches': mismatches,
            'content_sha256': manifest['content_sha256']}
