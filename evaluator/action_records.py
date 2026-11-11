"""Freeze original action outputs on both successful and failed attempts."""
import os
import uuid
from pathlib import Path

import file_tree
import integrity
import session


def archive(root, seq, action_dir):
    """Retain regular files without following links in runtime-generated trees."""
    destination = action_dir / 'artifacts'
    with session._session_lock(root):
        session._load_locked(root)
        captured = session._read(root / 'protection/manifest.json')['roots']
        if f'action_artifacts_{seq}' in captured:
            integrity.check_and_restore(root / 'protection', 'action_artifacts')
            return str((destination / 'index.json').relative_to(root))
    # Keep any previous attempt until the replacement is complete and durable.
    # The frozen workspace remains the recovery source on every failure path.
    temporary = action_dir / f'.artifacts.save_{uuid.uuid4().hex}'
    with file_tree.open_directory(temporary, create=True):
        pass
    inventory = []
    sources = [action_dir / name for name in (
        'result.json', 'stdout.log', 'stderr.log', 'execution.json', 'process_execution.json',
        'worker_diagnostics',
    )]
    sources.append(action_dir / 'workspace/.benchmark_outputs')
    scans = {}
    for source in sources:
        entries = file_tree.scan_tree(source, excluded_names=frozenset(), reject_special=False)
        scans[source] = entries
        for relative, entry in sorted(entries.items()):
            if source.name == '.benchmark_outputs' and entry['kind'] in ('symlink', 'special'):
                raise integrity.InfrastructureError('unsupported raw artifact; originals retained')
            if entry['kind'] == 'absent':
                continue
            if entry['kind'] == 'directory':
                directory = temporary / source.name / relative
                with file_tree.open_directory(directory, create=True):
                    pass
                continue
            source_file = source / relative if relative else source
            archive_path = Path(source.name) / relative if relative else Path(source.name)
            record = {'source': str(source_file.relative_to(root)), 'path': str(archive_path), **entry}
            if entry['kind'] == 'file':
                target = temporary / archive_path
                with file_tree.open_directory(target.parent, create=True):
                    pass
                with file_tree.open_file(source_file) as fd, file_tree.open_parent(target) as (parent, name):
                    output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444, dir_fd=parent)
                    try:
                        file_tree.copy_descriptor(fd, output)
                        os.fsync(output)
                    finally:
                        os.close(output)
                if file_tree.scan_tree(target)['']['sha256'] != entry['sha256']:
                    raise integrity.InfrastructureError('action artifact changed during archival')
            inventory.append(record)
    # Catch late files and removals, not just mutations of files already copied.
    for source, entries in scans.items():
        if entries != file_tree.scan_tree(source, excluded_names=frozenset(), reject_special=False):
            raise integrity.InfrastructureError('action artifact tree changed during archival')
    index = {'schema_version': 2, 'event_seq': seq,
             'candidate_snapshot': str((action_dir / 'workspace/.benchmark_context/starting_point').relative_to(root)),
             'files': inventory}
    file_tree.atomic_bytes(temporary / 'index.json', session._bytes(index), mode=0o444)
    # File fsync alone does not make newly created directory entries durable.
    for directory in sorted([temporary, *(p for p in temporary.rglob('*') if p.is_dir())],
                            key=lambda p: len(p.parts), reverse=True):
        with file_tree.open_directory(directory) as fd:
            os.fsync(fd)
    with file_tree.open_parent(destination) as (fd, name):
        file_tree.remove_at(fd, name)
        os.rename(temporary.name, name, src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)
    with session._session_lock(root):
        session._load_locked(root)
        integrity.register_roots(root / 'protection', {
            f'action_artifacts_{seq}': integrity.ProtectionRoot(destination, 'action_artifacts', readonly=True)})
    return str((destination / 'index.json').relative_to(root))


def verified_archive(root, index_path, *, event_seq=None):
    """Resolve a stable action identity only after verifying its protected files.

    No latest path, capture guess, or untrusted receipt grants authority.
    """
    relative = Path(index_path)
    if relative.is_absolute() or '..' in relative.parts or relative.name != 'index.json':
        raise integrity.InfrastructureError('invalid action artifact identity')
    destination = root / relative.parent
    with session._session_lock(root):
        session._load_locked(root)
        manifest = session._read(root / 'protection/manifest.json')['roots']
        index = session._read(root / relative)
        seq = index.get('event_seq')
        protected = manifest.get(f'action_artifacts_{seq}', {})
        if (index.get('schema_version') != 2 or type(seq) is not int
                or (event_seq is not None and seq != event_seq)
                or protected.get('path') != str(destination)):
            raise integrity.InfrastructureError('unregistered action artifact identity')
        integrity.check_and_restore(root / 'protection', 'action_artifacts')
        # Re-read after repair; the index itself is protected too.
        index = session._read(root / relative)
        for record in index['files']:
            if record['kind'] != 'file':
                continue
            path = Path(record['path'])
            if path.is_absolute() or '..' in path.parts:
                raise integrity.InfrastructureError('invalid archived artifact path')
            actual = file_tree.scan_tree(destination / path, excluded_names=frozenset())
            if actual.get('', {}).get('sha256') != record['sha256']:
                raise integrity.InfrastructureError('incomplete action artifact archive')
    return destination


def read_profile_archive(root, event_seq, *, file_ref=None, offset=0):
    """Bounded read-only public diagnostic export; never accepts a host path."""
    import base64
    import hashlib
    if type(event_seq) is not int or event_seq < 1 or type(offset) is not int or offset < 0:
        raise ValueError('invalid capture position')
    record = session.read_session(root)
    event = next((e for e in session._events(root, record) if e['seq'] == event_seq), None)
    if not event or event['action'] != 'profile' or event['mode'] not in session.PUBLIC_MODES:
        raise ValueError('not a registered public profile capture')
    saved = verified_archive(root, event['artifact_index'], event_seq=event_seq)
    mode = session.PUBLIC_MODES[event['mode']]
    base = saved / '.benchmark_outputs' / mode
    inventory = session._read(saved / 'index.json')['files']
    prefix = '.benchmark_outputs/' + mode + '/'
    files = []
    for item in inventory:
        if item['kind'] != 'file' or not item['path'].startswith(prefix):
            continue
        name = item['path'][len(prefix):]
        identity = hashlib.sha256(f"{record['session_id']}:{event_seq}:{name}:{item['sha256']}".encode()).hexdigest()
        path = base / name
        with file_tree.open_file(path) as fd:
            size = os.fstat(fd).st_size
        files.append({'file_ref': 'raw:' + identity, 'name': name, 'sha256': item['sha256'], 'size_bytes': size})
    files.sort(key=lambda f: f['name'])
    common = {'event_seq': event_seq, 'mode': event['mode'], 'readonly': True, 'hardware_actions_used': 0}
    def public_file(item):
        name = item['name']
        return {**item, 'name': name[:256], 'name_truncated': len(name) > 256}
    if file_ref is None:
        # Each file name can be large; cap the encoded page as well as row count.
        page = []
        for original in files[offset:offset + 32]:
            item = public_file(original)
            if page and len(session._bytes(page + [item])) > 4096:
                break
            page.append(item)
        following = offset + len(page)
        return {**common, 'files': page, 'next_offset': following if following < len(files) else None}
    item = next((f for f in files if f['file_ref'] == file_ref), None)
    if item is None:
        raise ValueError('file reference does not belong to this capture')
    if offset > item['size_bytes']:
        raise ValueError('file offset exceeds saved contents')
    with file_tree.open_file(base / item['name']) as fd:
        os.lseek(fd, offset, os.SEEK_SET)
        data = os.read(fd, 3072)
    following = offset + len(data)
    return {**common, **public_file(item), 'offset': offset, 'encoding': 'base64',
            'data': base64.b64encode(data).decode('ascii'),
            'next_offset': following if following < item['size_bytes'] else None}
