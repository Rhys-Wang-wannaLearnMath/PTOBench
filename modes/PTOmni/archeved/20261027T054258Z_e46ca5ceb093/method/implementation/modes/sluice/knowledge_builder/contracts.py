"""Strict internal and published contracts for the knowledge builder.

The builder deliberately keeps evidence and source identity in the private workspace.  Objects in
this module which are written to ``resources/knowledge`` contain semantic references only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json
import re


class ContractError(ValueError):
    pass


def _no_extra(value: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ContractError(f"unknown fields: {', '.join(sorted(unknown))}")


def _str_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise ContractError(f"{name} must be a list of non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class PublishedCard:
    id: str
    title: str
    domain: str
    summary: str
    evidence_status: str
    applicability: tuple[str, ...]
    principle: str
    source_features: tuple[str, ...]
    directions: tuple[str, ...]
    invariants: tuple[str, ...]
    effects: tuple[str, ...]
    risks: tuple[str, ...]
    questions: tuple[str, ...]
    requires: tuple[str, ...] = ()
    related: tuple[str, ...] = ()

    FIELDS = frozenset({
        "id", "title", "domain", "summary", "evidence_status", "applicability", "principle",
        "source_features", "directions", "invariants", "effects", "risks", "questions", "requires", "related",
    })

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PublishedCard":
        if not isinstance(raw, dict):
            raise ContractError("card must be an object")
        _no_extra(raw, set(cls.FIELDS))
        required = cls.FIELDS - {"requires", "related"}
        missing = required - set(raw)
        if missing:
            raise ContractError(f"card missing fields: {', '.join(sorted(missing))}")
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", raw["id"]):
            raise ContractError("invalid card id")
        if raw["evidence_status"] not in {"documented", "observed"}:
            raise ContractError("invalid evidence_status")
        if any(not isinstance(raw.get(k), str) or not raw[k].strip() for k in ("title", "domain", "summary", "principle")):
            raise ContractError("card text fields must be non-empty strings")
        values = {k: raw[k] for k in required}
        values["requires"] = raw.get("requires", [])
        values["related"] = raw.get("related", [])
        for k in ("applicability", "source_features", "directions", "invariants", "effects", "risks", "questions", "requires", "related"):
            values[k] = _str_list(values[k], k)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublishedReference:
    id: str
    title: str
    applicability: tuple[str, ...]
    body: str
    requires: tuple[str, ...] = ()

    FIELDS = frozenset({"id", "title", "applicability", "body", "requires"})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PublishedReference":
        if not isinstance(raw, dict):
            raise ContractError("reference must be an object")
        _no_extra(raw, set(cls.FIELDS))
        for k in ("id", "title", "body"):
            if not isinstance(raw.get(k), str) or not raw[k].strip():
                raise ContractError(f"reference {k} must be non-empty")
        return cls(raw["id"], raw["title"], _str_list(raw.get("applicability", []), "applicability"), raw["body"], _str_list(raw.get("requires", []), "requires"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublishedPack:
    id: str
    title: str
    summary: str
    keywords: tuple[str, ...]
    card_ids: tuple[str, ...]

    FIELDS = frozenset({"id", "title", "summary", "keywords", "card_ids"})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PublishedPack":
        if not isinstance(raw, dict):
            raise ContractError("pack must be an object")
        _no_extra(raw, set(cls.FIELDS))
        for k in ("id", "title", "summary"):
            if not isinstance(raw.get(k), str) or not raw[k].strip():
                raise ContractError(f"pack {k} must be non-empty")
        return cls(raw["id"], raw["title"], raw["summary"], _str_list(raw.get("keywords", []), "keywords"), _str_list(raw.get("card_ids", []), "card_ids"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobResult:
    job_id: str
    stage: str
    status: str
    output: Any = None
    error: str | None = None
    attempts: int = 0
    requests: tuple[dict[str, Any], ...] = ()

    def to_dict(self):
        return asdict(self)


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

