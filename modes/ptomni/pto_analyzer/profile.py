from __future__ import annotations

import hashlib
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class ProfileRefError(ValueError):
    """The supplied event reference cannot be bound to one valid profile."""


@dataclass(frozen=True)
class ProfileRef:
    session_root: Path
    session_id: str
    event_seq: int
    case_id: str
    candidate_sha256: str

    @property
    def event_id(self) -> str:
        return f"{self.session_id}:{self.event_seq}"

    @classmethod
    def from_event(cls, session_root: str | Path, event_seq: int) -> "ProfileRef":
        """Bind a caller-selected event; this does not search for a latest profile."""
        if not isinstance(session_root, (str, Path)):
            raise ProfileRefError("session root must be a path")
        try:
            root = Path(session_root).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProfileRefError("cannot resolve session root") from exc
        session = _read_json(root / "session.json", "session metadata")
        recorded_root = session.get("session_root")
        if recorded_root and (not isinstance(recorded_root, str) or Path(recorded_root).resolve() != root):
            raise ProfileRefError("session metadata points to a different session root")
        event = _find_event(root, event_seq)
        case_id = _event_case_id(event) or session.get("case_id", "")
        return cls(
            root,
            str(session.get("session_id", "")),
            event_seq,
            str(case_id),
            str(event.get("content_sha256", "")),
        )


@dataclass(frozen=True)
class ArtifactRead:
    path: str
    status: str
    sha256: str | None = None
    data: bytes | None = None
    error: str | None = None

    def source(self) -> dict[str, str]:
        ref = {"path": self.path}
        if self.sha256 is not None:
            ref["sha256"] = self.sha256
        return ref


class ProfileArchive:
    def __init__(self, ref: ProfileRef):
        try:
            self.root = Path(ref.session_root).resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ProfileRefError("ProfileRef has an invalid session root") from exc
        if not isinstance(ref.session_id, str) or not isinstance(ref.case_id, str):
            raise ProfileRefError("ProfileRef has invalid session or Case identity")
        session = _read_json(self.root / "session.json", "session metadata")
        if (
            session.get("session_id") != ref.session_id
            or session.get("case_id") != ref.case_id
            or session.get("session_root") and (
                not isinstance(session["session_root"], str)
                or Path(session["session_root"]).resolve() != self.root
            )
        ):
            raise ProfileRefError("ProfileRef does not match session metadata")

        event = _find_event(self.root, ref.event_seq)
        event_case_id = _event_case_id(event)
        if (
            event.get("action") != "profile"
            or event.get("mode") != "combined"
            or event.get("state") != "completed"
            or event.get("integrity_passed") is not True
            or (event_case_id is not None and event_case_id != ref.case_id)
            or event.get("content_sha256") != ref.candidate_sha256
        ):
            raise ProfileRefError("ProfileRef does not identify a completed combined profile")
        if not re.fullmatch(r"[0-9a-f]{64}", ref.candidate_sha256):
            raise ProfileRefError("ProfileRef has no valid Candidate digest")

        self.identity = {
            "session_id": ref.session_id,
            "event_seq": ref.event_seq,
            "event_id": ref.event_id,
            "case_id": ref.case_id,
            "candidate_sha256": ref.candidate_sha256,
            "mode": "combined",
        }
        self.artifact_root = self._safe_path(event.get("profile_artifacts_path"), "artifact root")
        index_path = self._safe_path(event.get("artifact_index"), "artifact index")
        index = _read_json(index_path, "artifact index")
        indexed_seq = index.get("event_seq")
        if type(indexed_seq) is not int or indexed_seq != ref.event_seq or not isinstance(index.get("files"), list):
            raise ProfileRefError("artifact index does not match the selected event")
        self._entries: dict[str, dict[str, Any]] = {}
        for entry in index["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ProfileRefError("artifact index contains an invalid file entry")
            rel = entry["path"]
            self._safe_relative(rel, "artifact path")
            if rel in self._entries:
                raise ProfileRefError("artifact index contains a duplicate path")
            self._entries[rel] = entry

        attempt_id = event.get("attempt_id")
        manifest_path = self.root / "protection" / "manifest.json"
        manifest = _read_json(manifest_path, "protection manifest")
        roots = manifest.get("roots")
        if not isinstance(roots, dict):
            raise ProfileRefError("protection manifest has no valid roots mapping")
        for root_name, protected_path in (("session", self.root / "session.json"), ("events", self.root / "events.jsonl")):
            root_record = roots.get(root_name)
            entries = root_record.get("entries") if isinstance(root_record, dict) else None
            entry = entries.get("") if isinstance(entries, dict) else None
            if not isinstance(entry, dict) or entry.get("kind") != "file" or _sha256_file(protected_path) != entry.get("sha256"):
                raise ProfileRefError(f"protected {root_name} identity does not match its manifest")
        root_record = roots.get(f"action_artifacts_{attempt_id}")
        if not isinstance(root_record, dict):
            raise ProfileRefError("protection manifest has an invalid artifact root")
        self._protected_entries = root_record.get("entries", {})
        if not isinstance(self._protected_entries, dict) or not self._protected_entries:
            raise ProfileRefError("protection manifest has an invalid artifact root")
        try:
            index_rel = index_path.relative_to(self.artifact_root).as_posix()
        except ValueError as exc:
            raise ProfileRefError("artifact index is outside the artifact root") from exc
        index_protection = self._protected_entries.get(index_rel)
        if (
            not isinstance(index_protection, dict)
            or index_protection.get("kind") != "file"
            or _sha256_file(index_path) != index_protection.get("sha256")
        ):
            raise ProfileRefError("artifact index does not match the protection manifest")
        self._cache: dict[str, ArtifactRead] = {}

    def artifact_paths(self, basename: str) -> list[str]:
        return sorted(path for path in self._entries if fnmatch.fnmatchcase(PurePosixPath(path).name, basename))

    def read(self, rel: str) -> ArtifactRead:
        if rel in self._cache:
            return self._cache[rel]
        entry = self._entries.get(rel)
        if entry is None:
            result = ArtifactRead(rel, "missing")
        else:
            expected = entry.get("sha256")
            protected = self._protected_entries.get(rel)
            if (
                entry.get("kind") != "file"
                or not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or not isinstance(protected, dict)
                or protected.get("kind") != "file"
                or protected.get("sha256") != expected
            ):
                result = ArtifactRead(rel, "invalid_data", error="index and protection manifest disagree")
            else:
                path = self._safe_path(self.artifact_root / rel, "archived artifact")
                try:
                    path.relative_to(self.artifact_root)
                except ValueError:
                    result = ArtifactRead(rel, "invalid_data", expected, error="archived file escapes artifact root")
                else:
                    try:
                        data = path.read_bytes()
                    except OSError as exc:
                        result = ArtifactRead(rel, "invalid_data", expected, error=f"archived file unavailable: {exc.__class__.__name__}")
                    else:
                        actual = hashlib.sha256(data).hexdigest()
                        if actual != expected:
                            result = ArtifactRead(rel, "invalid_data", actual, error="archived file hash mismatch")
                        elif not data:
                            result = ArtifactRead(rel, "empty", actual, data, "archived file is empty")
                        else:
                            result = ArtifactRead(rel, "present", actual, data)
        self._cache[rel] = result
        return result

    def sources(self, basename: str) -> list[ArtifactRead]:
        paths = self.artifact_paths(basename)
        return [self.read(path) for path in paths]

    def expected_source(self, basename: str) -> ArtifactRead:
        matches = self.sources(basename)
        return matches[0] if matches else ArtifactRead(f"<not indexed: {basename}>", "missing")

    def _safe_path(self, value: str | Path | None, label: str) -> Path:
        if value is None:
            raise ProfileRefError(f"missing {label}")
        try:
            path = Path(value)
            if not path.is_absolute():
                path = self.root / path
            resolved = path.resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ProfileRefError(f"invalid {label}") from exc
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ProfileRefError(f"{label} escapes the selected session") from exc
        return resolved

    @staticmethod
    def _safe_relative(value: str, label: str) -> PurePosixPath:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
            raise ProfileRefError(f"unsafe {label}")
        return path


def _find_event(root: Path, event_seq: int) -> dict[str, Any]:
    if not isinstance(event_seq, int) or isinstance(event_seq, bool) or event_seq < 1:
        raise ProfileRefError("event sequence must be a positive integer")
    try:
        lines = (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ProfileRefError("cannot read PTOBench event ledger") from exc
    matches = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProfileRefError("event ledger contains invalid JSON") from exc
        if isinstance(event, dict) and type(event.get("seq")) is int and event["seq"] == event_seq:
            matches.append(event)
    if len(matches) != 1:
        raise ProfileRefError("selected event is missing or ambiguous")
    return matches[0]


def _event_case_id(event: dict[str, Any]) -> Any:
    result = event.get("result")
    return event.get("case_id") or (result.get("case_id") if isinstance(result, dict) else None)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileRefError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise ProfileRefError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
