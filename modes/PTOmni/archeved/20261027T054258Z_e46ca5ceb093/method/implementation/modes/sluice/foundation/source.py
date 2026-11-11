"""Deterministic, method-side source manifest."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from .protocol import ProtocolError


_EXCLUDED = {".benchmark_context", ".benchmark_outputs", ".git", ".pytest_cache", "__pycache__"}


@dataclass(frozen=True)
class SourceFingerprint:
    files: tuple[dict[str, str], ...]
    directories: tuple[str, ...]
    content_sha256: str

    @property
    def content_hash(self) -> str:
        return self.content_sha256

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "algorithm": "sluice-source-manifest-v1",
                "files": [dict(item) for item in self.files], "directories": list(self.directories),
                "content_sha256": self.content_sha256}


def _digest_file(path: Path) -> str:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ProtocolError(f"unsupported source entry: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except (OSError, ValueError) as exc:
        raise ProtocolError("source file could not be read", "infrastructure_error") from exc
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ProtocolError("source changed while it was fingerprinted", "infrastructure_error")
    return "sha256:" + digest.hexdigest()


def fingerprint(root: str | Path) -> SourceFingerprint:
    base = Path(root)
    if not base.is_absolute() or base.is_symlink() or not base.is_dir():
        raise ProtocolError("source root is invalid")
    files: list[dict[str, str]] = []
    directories: list[str] = ["."]
    for current, dirnames, filenames in os.walk(base, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            if name in _EXCLUDED:
                continue
            path = current_path / name
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise ProtocolError("source entry could not be inspected", "infrastructure_error") from exc
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ProtocolError("source contains an unsupported entry")
            kept_dirs.append(name)
            rel = path.relative_to(base).as_posix()
            directories.append(rel)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            if name in _EXCLUDED:
                continue
            path = current_path / name
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise ProtocolError("source entry could not be inspected", "infrastructure_error") from exc
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ProtocolError("source contains an unsupported entry")
            files.append({"path": path.relative_to(base).as_posix(), "sha256": _digest_file(path)})
    files.sort(key=lambda item: item["path"])
    directories = sorted(set(directories))
    canonical = "".join(f"{item['path']}\0{item['sha256']}\n" for item in files).encode()
    total = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return SourceFingerprint(tuple(files), tuple(directories), total)
