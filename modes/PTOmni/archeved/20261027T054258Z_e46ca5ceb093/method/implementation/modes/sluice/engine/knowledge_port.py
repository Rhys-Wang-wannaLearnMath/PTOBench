"""Source-verified, bounded local adapted-skill retrieval; replaceable by 03."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from .contracts import KnowledgeDocument, KnowledgeHit
from ..foundation.protocol import ProtocolError
from json import loads


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def safe_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ProtocolError("Invalid resource reference")
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ProtocolError("Resource links are not allowed")
    if not current.is_file():
        raise ProtocolError("Resource is missing")
    return current


def bound_knowledge(root, *, validated_environment_digest=None):
    """Load the external release binding; metadata never enters tool responses."""
    root = Path(root)
    try:
        binding = json.loads(safe_file(root.parent, root.name + "_release.json").read_text(encoding="utf-8"))
        return PublishedKnowledgePort(
            root, binding, validated_environment_digest=validated_environment_digest
        )
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError("Knowledge release binding is unavailable or invalid") from exc


def lint_active(text):
    """Reject legacy executable control and hidden/raw paths in exposed instructions."""
    patterns = [
        r"\b(?:state_transition|complete_stage|complete_phase|todowrite|todoread)\b",
        r"\b(?:merged_swimlane\.json|bubble_analysis\.log|machine_runtime_operator_trace)\b",
        r"(?:/root/|/home/|sealed_assets|\.benchmark_outputs|evaluator/)",
        r"\b(?:Task|Grep|Glob|WebFetch)\s*\(",
        r"\bpython\d?\s+\S+\.py",
        r"(?im)^\s*(?:[-*]\s*)?(?:read|parse|open|load)\s+(?:the\s+)?(?:complete\s+|full\s+)?raw(?:[ -]profile)?(?:\s+JSON)?\b",
        r"(?:^|[\s`])\.(?!benchmark_context/)[A-Za-z_][A-Za-z0-9_-]*/",
        r"(?:读取|解析|打开).{0,15}(?:原始|完整).{0,8}(?:profile|profiling|JSON)",
        r"\bS[1-5]_(?:SETUP|COLLECT|ANALYZE|TUNE|REPORT)\b",
    ]
    return [p for p in patterns if re.search(p, text)]


class LocalSkillKnowledgePort:
    version = "sluice-local-catalog-v3-pure"

    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads(safe_file(root, "skill_manifest.json").read_text())
        if self.manifest.get("schema_version") != 2:
            raise ProtocolError("Unsupported skill manifest version")
        lockpath = safe_file(root, "upstream.lock.json")
        if sha(lockpath) != self.manifest["upstream"]["lock_sha256"]:
            raise ProtocolError("Upstream lock hash mismatch")
        lock = json.loads(lockpath.read_text())
        if set(lock["files"]) != {e["source"] for e in self.manifest["entries"]}:
            raise ProtocolError("Incomplete skill inventory")
        self.active = {}
        for e in self.manifest["entries"]:
            if e["source_sha256"] != "sha256:" + lock["files"][e["source"]]:
                raise ProtocolError("Source is not locked")
            if not e["active"]:
                continue
            if (
                e["disposition"] != "adapt"
                or e["status"] != "reviewed"
                or not e["reason"]
            ):
                raise ProtocolError("Unreviewed active skill")
            self._verify(e)
            self.active[e["id"]] = e
        for filename, key in (
            ("compatibility.json", "compatibility_sha256"),
            ("mechanism_review.json", "mechanism_review_sha256"),
        ):
            if sha(safe_file(root, filename)) != self.manifest[key]:
                raise ProtocolError("Knowledge review hash mismatch")
        self.compatibility = json.loads(
            safe_file(root, "compatibility.json").read_text()
        )
        if (
            sha(safe_file(root, "catalog_index.json"))
            != self.manifest["catalog_index_sha256"]
        ):
            raise ProtocolError("Catalog index hash mismatch")
        self.index = json.loads((root / "catalog_index.json").read_text())
        if any(m["skill"] not in self.active for m in self.index):
            raise ProtocolError("Unresolvable mechanism skill")

    def _verify(self, entry):
        target = safe_file(self.root, entry["target"])
        if sha(target) != entry["target_sha256"]:
            raise ProtocolError("Skill content hash mismatch")
        text = target.read_text()
        if (
            entry["source"] not in text
            or entry["source_sha256"] not in text
            or lint_active(text)
        ):
            raise ProtocolError("Active skill instruction/provenance check failed")
        return target

    def _ref(self, entry):
        return "card:" + entry["id"]

    def search(self, *, case_id, source_hash, query, observations, limit=5):
        words = set(
            re.findall(r"[\w-]+", query.lower() + " " + " ".join(observations).lower())
        )
        requested_ids = set(re.findall(r"\b[FSI]-\d+\b", query.upper()))
        ranked = []
        for e in self.active.values():
            body = self._verify(e).read_text().lower()
            mechanisms = [m for m in self.index if m["skill"] == e["id"]]
            mechanisms.sort(
                key=lambda m: (
                    m["id"] not in requested_ids,
                    -sum(w in m["title"].lower() for w in words),
                    m["id"],
                )
            )
            summary = "; ".join(m["id"] + " " + m["title"] for m in mechanisms)
            score = (
                100 * sum(m["id"] in requested_ids for m in mechanisms)
                + 3
                * sum(w in (e["id"] + " " + e["description"]).lower() for w in words)
                + 4
                * sum(any(w in m["title"].lower() for m in mechanisms) for w in words)
                + sum(min(body.count(w), 6) for w in words)
            )
            hit = KnowledgeHit(
                self._ref(e),
                e["id"],
                summary[:1800],
                tuple(m["preconditions"] for m in mechanisms[:3])
                or ("Check the applicable semantic constraints",),
                tuple(mechanisms[0]["expected_signals"])
                if mechanisms
                else ("clean_latency_improvement",),
                tuple(mechanisms[0]["risks"])
                + ("API availability does not grant tuning permission",)
                if mechanisms
                else ("Check the permitted capabilities",),
            )
            if score > 0:
                ranked.append((-score, e["id"], hit))
        return [v[2] for v in sorted(ranked)[:limit]]

    def read(self, ref, *, max_bytes=12000, offset=0, paginate=False):
        kind, sep, name = ref.partition(":")
        if kind != "card":
            raise ProtocolError("Unknown knowledge reference")
        e = self.active.get(name)
        if not e or ref != self._ref(e): raise ProtocolError("Unknown knowledge reference")
        self._verify(e)
        if not 1 <= max_bytes <= 32768 or offset != 0 or paginate:
            raise ProtocolError("Invalid complete knowledge read bound")
        # The legacy file is audited internally; only organized catalog semantics leave this port.
        fields = ('id', 'title', 'preconditions', 'expected_signals', 'risks', 'rollback_condition')
        mechanisms = [{key: item[key] for key in fields} for item in self.index if item['skill'] == name]
        content = json.dumps({'title': name, 'mechanisms': mechanisms}, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        required = len(content.encode('utf-8'))
        if required > max_bytes:
            return KnowledgeDocument(ref, '', 0, True, needs_more_budget=True, required_bytes=required)
        return KnowledgeDocument(ref, content, 0, False, required_bytes=required)

    def contains(self, ref):
        kind, sep, name = ref.partition(":")
        return kind == "card" and bool(sep) and name in self.active


class PublishedKnowledgePort:
    """Read-only adapter for the four-directory pure knowledge package."""
    version = "sluice-published-knowledge-v1"
    def __init__(self, root: Path, binding: dict | None = None, *, validated_environment_digest=None):
        self.root = Path(root)
        if not (self.root / "cards").is_dir() or not (self.root / "reference").is_dir() or not (self.root / "packs").is_dir() or not (self.root / "index/search.json").is_file():
            raise ProtocolError("Knowledge package is unavailable")
        try:
            # Reuse the builder's strict schema/reference closure gate at the runtime boundary.
            from ..knowledge_builder.publish import validate_package
            validate_package(self.root)
            from ..knowledge_builder.review import static_review
            if static_review(self.root)['decision'] != 'accept':
                raise ValueError('knowledge purity review failed')
        except Exception as exc:
            raise ProtocolError("Knowledge package is invalid") from exc
        self.cards = {p.stem: loads(p.read_text(encoding="utf-8")) for p in (self.root / "cards").glob("*.json")}
        self.refs = {p.stem: loads(p.read_text(encoding="utf-8")) for p in (self.root / "reference").glob("*.json")}
        self.index = loads((self.root / "index/search.json").read_text(encoding="utf-8"))
        self.mechanism_bindings = loads((self.root / "index/mechanism_bindings.json").read_text(encoding="utf-8"))['bindings']
        if binding is not None:
            try:
                from ..knowledge_builder.environment import collect_interpreter, release_environment_identity
                from ..knowledge_builder.publish import package_fingerprint
                if not all(isinstance(binding.get(key), str) and binding[key] for key in ('environment', 'knowledge_digest', 'python_executable')):
                    raise ValueError('incomplete binding')
                if package_fingerprint(self.root) != binding['knowledge_digest']:
                    raise ValueError('content differs')
                if validated_environment_digest is None:
                    actual = release_environment_identity(collect_interpreter(binding['python_executable']))
                    identity = hashlib.sha256(json.dumps(actual, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
                else:
                    if (not isinstance(validated_environment_digest, str)
                            or not re.fullmatch(r'[0-9a-f]{64}', validated_environment_digest)):
                        raise ValueError('invalid validated environment digest')
                    identity = validated_environment_digest
                if identity != binding['environment']:
                    raise ValueError('environment differs')
            except Exception as exc:
                raise ProtocolError('Knowledge package environment or content binding mismatch') from exc

        from ..knowledge_builder.retrieval import KnowledgeStore
        self._store = KnowledgeStore(self.root)
        self.bound = binding is not None
        self.binding = dict(binding) if binding is not None else None
    def contains(self, ref):
        kind, sep, name = ref.partition(":")
        return bool(sep) and ((kind == "card" and name in self.cards) or (kind == "ref" and name in self.refs))
    def search(self, *, case_id, source_hash, query, observations, limit=5):
        page=self.search_page(query=query,observations=observations,
                              limit=max(1,min(8,limit)))
        return page['items']
    def search_page(self, *, query='', observations=(), limit=8, topic=None, cursor=None, browse=False):
        page = self._store.search_page(query, observations=observations, limit=limit,
                                       topic=topic, cursor=cursor, browse=browse)
        items=[]
        for item in page['items']:
            signals=tuple(item.get('expected_signals', ()))
            bindings=tuple({'question_id': evidence['question_id'],
                            'signal_keys': tuple(evidence['signal_keys'])}
                for binding in self.mechanism_bindings if binding['card_ref']==item['id']
                for evidence in binding['evidence'])
            mapping=('Registered evidence questions: ' + ', '.join(
                dict.fromkeys(evidence['question_id'] for evidence in bindings))
                     if bindings else 'No registered observation mapping is available.')
            items.append(KnowledgeHit(item['id'], item.get('title',''), item.get('summary',''),
                preconditions=tuple(item.get('preconditions', ())), expected_signals=signals,
                signal_mapping=mapping, evidence_bindings=bindings,
                risks=tuple(item.get('risks', ()))))
        return {'items':items, 'next_cursor':page['next_cursor'],
                'total_matches':page['total_matches'], 'topics':page['topics']}
    def read(self, ref, *, max_bytes=12000, offset=0, paginate=False):
        from ..knowledge_builder.retrieval import KnowledgeStore
        item = self._store.read(ref, max_bytes, offset, paginate=paginate)
        return KnowledgeDocument(ref, content=item["content"], offset=item.get("offset", 0), truncated=item.get("truncated", False), needs_more_budget=item.get("needs_more_budget", False), required_bytes=item.get("required_bytes"))
    def read_page(self, ref, *, max_bytes=12000, cursor=None):
        item=self._store.read_page(ref,max_bytes,cursor=cursor)
        return KnowledgeDocument(ref, content=item['content'], offset=item.get('offset',0),
            truncated=item['truncated'], needs_more_budget=item['needs_more_budget'],
            required_bytes=item['required_bytes'], requires=tuple(item['requires']),
            related=tuple(item['related']), dependencies=tuple(item['dependencies']),
            next_cursor=item['next_cursor'], complete=item['complete'],
            include_page_fields=True)
