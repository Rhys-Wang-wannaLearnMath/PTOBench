"""Freeze original action outputs on both successful and failed attempts."""
import os
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
    # A process may die between writing files and publishing their inventory.
    with file_tree.open_parent(destination) as (fd, name):
        file_tree.remove_at(fd, name)
    with file_tree.open_directory(destination, create=True):
        pass
    inventory = []
    sources = [action_dir / name for name in ('result.json', 'stdout.log', 'stderr.log', 'execution.json', 'process_execution.json')]
    sources.append(action_dir / 'workspace/.benchmark_outputs')
    for source in sources:
        entries = file_tree.scan_tree(source, reject_special=False)
        for relative, entry in sorted(entries.items()):
            if entry['kind'] in ('directory', 'absent'):
                continue
            source_file = source / relative if relative else source
            archive_path = Path(source.name) / relative if relative else Path(source.name)
            record = {'source': str(source_file.relative_to(root)), 'path': str(archive_path), **entry}
            if entry['kind'] == 'file':
                target = destination / archive_path
                with file_tree.open_directory(target.parent, create=True):
                    pass
                with file_tree.open_file(source_file) as fd, file_tree.open_parent(target) as (parent, name):
                    output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444, dir_fd=parent)
                    try:
                        file_tree.copy_descriptor(fd, output)
                    finally:
                        os.close(output)
                if file_tree.scan_tree(target)['']['sha256'] != entry['sha256']:
                    raise integrity.InfrastructureError('action artifact changed during archival')
            inventory.append(record)
    index = {'schema_version': 1, 'event_seq': seq,
             'candidate_snapshot': str((action_dir / 'workspace/.benchmark_context/starting_point').relative_to(root)),
             'files': inventory}
    file_tree.atomic_bytes(destination / 'index.json', session._bytes(index), mode=0o444)
    with session._session_lock(root):
        session._load_locked(root)
        integrity.register_roots(root / 'protection', {
            f'action_artifacts_{seq}': integrity.ProtectionRoot(destination, 'action_artifacts', readonly=True)})
    return str((destination / 'index.json').relative_to(root))
