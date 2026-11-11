"""Deterministic public evidence serialization and recursive disclosure guard."""
from __future__ import annotations
import hashlib
import json
import math
import re

_FORBIDDEN = re.compile(
    r'candidate|hidden|sealed|oracle|evaluator|traceback|duckdb|sha256|'
    r'sk-[A-Za-z0-9]|api[_-]?key|bearer\s|[/\\]|'
    r'\b(?:action_id|event_seq|run_id|artifact_dir|locator|records|deps|sql)\b|'
    r'[0-9a-f]{32,}|(?:^|[\s"\'])/(?:[^\s"\']+)|[A-Za-z]:\\|\.\.[/\\]', re.I)
_REF = re.compile(r'^(?:profile|evidence|hypothesis|program|revision):[a-z2-7]{16,32}$')
_NATIVE_FORBIDDEN = re.compile(_FORBIDDEN.pattern.replace('candidate|', ''), re.I)


def opaque(kind, value):
    import base64
    token = base64.b32encode(hashlib.sha256(str(value).encode()).digest()[:12]).decode().rstrip('=').lower()
    return kind + ':' + token


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def scan(value):
    if isinstance(value, dict):
        for key, item in value.items():
            scan(key)
            if key in ('message', 'source_file') and isinstance(item, str):
                # Native compiler text may legitimately mention Candidate.
                # Paths, private identifiers and secrets retain the same guard.
                if len(item) > 512 or _NATIVE_FORBIDDEN.search(item):
                    raise ValueError('public_evidence_rejected')
            else:
                scan(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            scan(item)
    elif isinstance(value, str):
        # Opaque base32 tokens can accidentally spell a forbidden word. Their
        # fixed typed grammar carries no path, SQL or arbitrary backend text;
        # refusing a valid identifier would break pages and historical reads.
        if _REF.fullmatch(value):
            return
        if len(value) > 512 or _FORBIDDEN.search(value):
            raise ValueError('public_evidence_rejected')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('public_evidence_rejected')
    elif value is not None and type(value) not in (int, float, bool):
        raise ValueError('public_evidence_rejected')


def require_ref(value, kind):
    if not isinstance(value, str) or not value.startswith(kind + ':') or not _REF.fullmatch(value):
        raise ValueError('public_reference_rejected')
    return value
