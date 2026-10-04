"""S6 evidence: hash every canonical byte, including large tensor files."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

bench = Path(__file__).resolve().parents[3]
destination = Path(sys.argv[1])
entries = {}
total = 0
started = time.monotonic()
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def hash_entry(path):
    relative = path.relative_to(bench).as_posix()
    if path.is_symlink():
        return relative, {'kind': 'symlink', 'target': os.readlink(path)}
    if path.is_dir():
        return relative, {'kind': 'directory'}
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as reader:
        while chunk := reader.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    return relative, {'kind': 'file', 'size': after.st_size, 'sha256': digest.hexdigest(),
                      'stable_during_read': (before.st_ino, before.st_size, before.st_mtime_ns)
                      == (after.st_ino, after.st_size, after.st_mtime_ns)}


with ThreadPoolExecutor(max_workers=workers) as pool:
    for group in ('cases', 'sealed_assets', 'environments'):
        paths = [Path(parent) / name for parent, dirs, files in os.walk(bench / group, followlinks=False)
                 for name in sorted(dirs + files)]
        for relative, value in pool.map(hash_entry, paths):
            entries[relative] = value
            total += value.get('size', 0)
            if len(entries) % 1000 == 0:
                print(f'{group}: {len(entries)} entries, {total / 1024**3:.2f} GiB', flush=True)
payload = {'algorithm': 'sha256', 'entries': entries, 'bytes': total,
           'elapsed_seconds': time.monotonic() - started, 'workers': workers}
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'Finished {len(entries)} entries / {total / 1024**3:.2f} GiB in {payload["elapsed_seconds"]:.1f}s', flush=True)
