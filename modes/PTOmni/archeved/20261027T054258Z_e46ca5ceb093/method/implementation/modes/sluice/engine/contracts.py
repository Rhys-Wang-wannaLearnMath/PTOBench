"""Versioned JSON contracts. No evaluator or future database implementation imports."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence
import hashlib
import json


def digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )


@dataclass(frozen=True)
class Contract:
    schema_version: int = field(default=1, init=False)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), ensure_ascii=False, allow_nan=False))


@dataclass(frozen=True, init=False)
class KnowledgeHit(Contract):
    """Pure search result.  Source identity is intentionally absent."""
    ref: str
    title: str
    summary: str
    preconditions: tuple[str, ...] = ()
    expected_signals: tuple[str, ...] = ()
    signal_mapping: str = "No registered observation mapping is available."
    evidence_bindings: tuple[dict, ...] = ()
    risks: tuple[str, ...] = ()
    rollback_condition: str = "Correctness failure or latency regression"

    def __init__(self, ref, title, summary, *legacy, preconditions=(), expected_signals=(), signal_mapping="No registered observation mapping is available.", evidence_bindings=(), risks=(), rollback_condition="Correctness failure or latency regression"):
        # Accept the old positional source/hash pair for workspace compatibility, but never retain or serialize it.
        if legacy and isinstance(legacy[0], (tuple, list)):
            preconditions, expected_signals, risks = legacy[0], (legacy[1] if len(legacy)>1 else expected_signals), (legacy[2] if len(legacy)>2 else risks)
        elif len(legacy) >= 2 and isinstance(legacy[0], str):
            preconditions = legacy[2] if len(legacy)>2 else preconditions
            expected_signals = legacy[3] if len(legacy)>3 else expected_signals
            risks = legacy[4] if len(legacy)>4 else risks
        object.__setattr__(self, "schema_version", 1); object.__setattr__(self, "ref", ref); object.__setattr__(self, "title", title); object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "preconditions", tuple(preconditions)); object.__setattr__(self, "expected_signals", tuple(expected_signals)); object.__setattr__(self, "signal_mapping", signal_mapping); object.__setattr__(self, "evidence_bindings", tuple(dict(item) for item in evidence_bindings)); object.__setattr__(self, "risks", tuple(risks)); object.__setattr__(self, "rollback_condition", rollback_condition)

    def to_dict(self):
        value = {
            'schema_version': self.schema_version, 'ref': self.ref, 'title': self.title,
            'summary': self.summary, 'preconditions': list(self.preconditions),
            'expected_signals': list(self.expected_signals), 'risks': list(self.risks),
            'rollback_condition': self.rollback_condition,
            'signal_mapping': self.signal_mapping,
            'evidence_bindings': [dict(item) for item in self.evidence_bindings],
        }
        return value


@dataclass(frozen=True, init=False)
class KnowledgeDocument(Contract):
    """Pure readable knowledge document; no original path or digest."""
    ref: str
    content: str
    offset: int = 0
    truncated: bool = False
    needs_more_budget: bool = False
    required_bytes: int | None = None
    requires: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    next_cursor: str | None = None
    complete: bool = True

    def __init__(self, ref, *args, content=None, offset=0, truncated=False, needs_more_budget=False, required_bytes=None, requires=(), related=(), dependencies=(), next_cursor=None, complete=True, include_page_fields=False):
        # Legacy calls were (ref, source, source_sha256, content, offset, truncated).
        if content is None:
            if len(args) >= 3 and isinstance(args[2], str): content = args[2]; offset = args[3] if len(args)>3 else 0; truncated = args[4] if len(args)>4 else False
            elif args: content = args[0]; offset = args[1] if len(args)>1 else 0; truncated = args[2] if len(args)>2 else False
            else: content = ""
        object.__setattr__(self, "schema_version", 1); object.__setattr__(self, "ref", ref); object.__setattr__(self, "content", content); object.__setattr__(self, "offset", offset); object.__setattr__(self, "truncated", truncated); object.__setattr__(self, "needs_more_budget", needs_more_budget); object.__setattr__(self, "required_bytes", required_bytes)
        object.__setattr__(self, "requires", tuple(requires)); object.__setattr__(self, "related", tuple(related)); object.__setattr__(self, "dependencies", tuple(dependencies)); object.__setattr__(self, "next_cursor", next_cursor); object.__setattr__(self, "complete", complete)
        object.__setattr__(self, "_extended", bool(include_page_fields or requires or related or dependencies or next_cursor is not None or not complete))

    def to_dict(self):
        value = {'schema_version': self.schema_version, 'ref': self.ref, 'content': self.content,
                 'offset': self.offset, 'truncated': self.truncated,
                 'needs_more_budget': self.needs_more_budget, 'required_bytes': self.required_bytes}
        if getattr(self, '_extended', False):
            value.update(requires=list(self.requires), related=list(self.related),
                         dependencies=list(self.dependencies), next_cursor=self.next_cursor,
                         complete=self.complete)
        return value


@dataclass(frozen=True)
class Hypothesis(Contract):
    hypothesis_id: str
    case_id: str
    parent_source_hash: str
    mechanism: str | None
    change_scope: tuple[str, ...]
    preconditions: tuple[str, ...]
    expected_signals: tuple[str, ...]
    verification_questions: tuple[str, ...]
    rollback_condition: str
    knowledge_refs: tuple[str, ...]
    status: str = "registered"
    opportunity_id: str | None = None
    mechanism_id: str | None = None
    program_ref: str | None = None
    kind: str = "optimize"
    schema_version: int = field(default=2, init=False)

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported contract schema version")
        value = {k: v for k, v in value.items() if k != "schema_version"}
        for k in (
            "change_scope",
            "preconditions",
            "expected_signals",
            "verification_questions",
            "knowledge_refs",
        ):
            value[k] = tuple(value[k])
        value.setdefault('opportunity_id', None)
        value.setdefault('mechanism_id', None)
        value.setdefault('program_ref', None)
        value.setdefault('kind', 'optimize')
        return cls(**value)

    def to_dict(self):
        value = super().to_dict()
        value['schema_version'] = 2
        return value


@dataclass(frozen=True)
class ProfileRef(Contract):
    ref: str
    case_id: str
    source_hash: str
    action_id: str
    event_seq: int | None
    mode: str
    observation_ref: str
    availability: str = "available"
    reason: str | None = None
    members: tuple[str, ...] = ()
    member_kind: str | None = None
    member_slots: tuple[str, ...] = ()

    def to_dict(self):
        from .evidence_guard import scan
        value = dict(schema_version=2, ref=self.ref, mode=self.mode, availability=self.availability)
        if self.reason:
            value['reason'] = self.reason
        limit = 4
        if self.members:
            value['members'] = list(self.members[:limit])
        if self.member_kind:
            value['member_kind'] = self.member_kind
        if self.member_slots:
            value['member_slots'] = dict(zip(self.member_slots[:limit], self.members[:limit]))
        if len(self.members) > limit:
            value.update(member_count=len(self.members), members_truncated=True, member_detail_view='members')
        scan(value)
        return value


@dataclass(frozen=True)
class EvidenceFact(Contract):
    question: str
    key: str
    value: Any
    unit: str | None
    status: str
    source: str
    reason: str
    action_id: str | None = None

    def to_dict(self):
        from .evidence_guard import require_ref, scan
        value = dict(schema_version=2, question=self.question, key=self.key, value=self.value,
                     unit=self.unit, status=self.status, source_ref=require_ref(self.source, 'evidence'), reason=self.reason)
        if self.status not in ('measured', 'derived', 'unavailable'):
            raise ValueError('invalid_evidence_status')
        if isinstance(self.value, dict) or isinstance(self.value, (tuple, list)) and (
            len(self.value) > 16 or any(isinstance(v, (dict, list, tuple)) for v in self.value)):
            raise ValueError('invalid_evidence_value')
        scan(value)
        return value


@dataclass(frozen=True)
class EvidenceBundle(Contract):
    question: str
    facts: tuple[EvidenceFact, ...] = ()
    status: str = "unavailable"
    missing: tuple[str, ...] = ()
    comparison: str = "unavailable"
    truncated: bool = False
    provenance: tuple[str, ...] = ()
    evidence_ref: str = ""
    mechanism_id: str | None = None
    hypothesis_ref: str = ""
    program_ref: str = ""

    def to_json(self):
        from .evidence_guard import encoded
        return encoded(self.to_dict()).decode()

    def to_dict(self):
        from .evidence_guard import opaque, require_ref, scan
        ref = self.evidence_ref or opaque('evidence', self.question)
        value = dict(schema_version=2, evidence_ref=require_ref(ref, 'evidence'), question=self.question,
                     mechanism_id=self.mechanism_id, hypothesis_ref=self.hypothesis_ref or opaque('hypothesis', ref),
                     program_ref=self.program_ref or opaque('program', ref), facts=[f.to_dict() for f in self.facts],
                     status=self.status, comparison=self.comparison, missing=list(self.missing),
                     truncated=self.truncated, provenance=list(self.provenance))
        require_ref(value['hypothesis_ref'], 'hypothesis')
        require_ref(value['program_ref'], 'program')
        for item in self.provenance:
            require_ref(item, 'profile')
        if self.comparison not in ('improved', 'regressed', 'decreased', 'increased', 'changed',
                                   'no_change', 'inconclusive', 'incomparable', 'unavailable'):
            raise ValueError('invalid_evidence_comparison')
        if self.status not in ('available', 'unavailable'):
            raise ValueError('invalid_evidence_status')
        scan(value)
        return value

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported contract schema version")
        if value.get('schema_version') == 1:
            from .evidence_guard import opaque
            value = dict(value)
            ref = opaque('evidence', value['question'])
            facts = []
            for old in value.get('facts', []):
                status = {'proven': 'derived', 'unproven': 'unavailable'}.get(old['status'], old['status'])
                facts.append(dict(question=old['question'], key=old['key'],
                    value=old['value'] if status != 'unavailable' else None, unit=old.get('unit'),
                    status=status, source=ref, reason='legacy_evidence'))
            value.update(facts=facts, comparison='unavailable',
                status='available' if any(f['status'] != 'unavailable' for f in facts) else 'unavailable',
                provenance=[opaque('profile', p) for p in value.get('provenance', [])])
        value = {k: v for k, v in value.items() if k != "schema_version"}
        value["facts"] = tuple(
            EvidenceFact(**{('source' if k == 'source_ref' else k): v for k, v in f.items() if k != "schema_version"}) for f in value["facts"]
        )
        for k in ("missing", "provenance"):
            value[k] = tuple(value[k])
        return cls(**value)


@dataclass(frozen=True)
class Attempt(Contract):
    attempt_id: str
    hypothesis_id: str
    case_id: str
    parent_source_hash: str
    child_source_hash: str
    action_id: str
    event_seq: int | None
    observation_ref: str
    outcome: str
    correctness: str | None
    latency_us: float | None
    evidence_status: str = "not_requested"

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported contract schema version")
        return cls(**{k: v for k, v in value.items() if k != "schema_version"})


@dataclass(frozen=True)
class ProgressView(Contract):
    revision: str
    active_hypothesis: Hypothesis | None = None
    attempts: tuple[Attempt, ...] = ()
    best: Attempt | None = None
    evidence: tuple[EvidenceBundle, ...] = ()
    provisional: bool = True

    # v2 projections.  These fields are views over the same event journal;
    # they are intentionally optional so frozen v1 journals remain readable.
    schema_version: int = field(default=2, init=False)
    programs: tuple["ProgramVersion", ...] = ()
    working_program_ref: str | None = None
    last_verified_program_ref: str | None = None
    best_verified_program_ref: str | None = None
    cycles: tuple["OptimizationCycle", ...] = ()
    opportunities: tuple["Opportunity", ...] = ()
    decision: "Decision | None" = None
    state_revision: str = ""

    def to_dict(self):
        value = super().to_dict()
        value['evidence'] = [bundle.to_dict() for bundle in self.evidence]
        value['programs'] = [item.to_dict() for item in self.programs]
        value['cycles'] = [item.to_dict() for item in self.cycles]
        value['opportunities'] = [item.to_dict() for item in self.opportunities]
        value['decision'] = self.decision.to_dict() if self.decision else None
        value['schema_version'] = 2
        return value


@dataclass(frozen=True)
class Decision(Contract):
    action: str
    reason: str
    hypothesis_id: str | None = None
    opportunity_id: str | None = None
    evidence_question: str | None = None
    expected_revision: str | None = None
    proposed_by: str = "controller"
    schema_version: int = field(default=2, init=False)

    ACTIONS = frozenset({"optimize", "diagnose", "recheck", "stop"})

    def __post_init__(self):
        if self.action not in self.ACTIONS:
            raise ValueError("unsupported decision action")
        if not self.reason or len(self.reason) > 2000:
            raise ValueError("decision reason is required")

    def to_dict(self):
        value = super().to_dict()
        value['schema_version'] = 2
        return value


@dataclass(frozen=True)
class ProgramVersion(Contract):
    program_ref: str
    case_id: str
    source_hash: str
    parent_program_ref: str | None = None
    source_ref: str | None = None
    action_id: str | None = None
    snapshot_id: str | None = None
    verification: str = "unverified"
    latency_us: float | None = None
    lineage: tuple[str, ...] = ()
    schema_version: int = field(default=2, init=False)

    def __post_init__(self):
        if self.verification not in {"seed", "working", "unverified", "verified", "best", "failed", "dirty"}:
            raise ValueError("invalid program verification status")


@dataclass(frozen=True)
class OptimizationCycle(Contract):
    cycle_id: str
    kind: str
    program_ref: str
    parent_program_ref: str | None
    hypothesis_id: str | None
    action_id: str | None
    result: str
    evidence_refs: tuple[str, ...] = ()
    status: str = "completed"
    unknowns: tuple[str, ...] = ()
    schema_version: int = field(default=2, init=False)

    def __post_init__(self):
        if self.kind not in {"optimize", "diagnose", "recheck"}:
            raise ValueError("invalid optimization cycle kind")
        if self.status not in {"pending", "collecting_evidence", "feedback_ready", "completed", "terminal"}:
            raise ValueError("invalid optimization cycle status")


@dataclass(frozen=True)
class Assessment(Contract):
    explanation: str
    refs: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    schema_version: int = field(default=2, init=False)


@dataclass(frozen=True)
class Opportunity(Contract):
    opportunity_id: str
    mechanism_id: str | None
    source_scope: tuple[str, ...]
    status: str = "open"
    assessment: dict[str, Assessment] = field(default_factory=dict)
    cycle_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    condition_change: str | None = None
    schema_version: int = field(default=2, init=False)

    STATUSES = frozenset({"open", "needs_evidence", "blocked", "improved", "recheck", "closed"})

    def __post_init__(self):
        if self.status not in self.STATUSES:
            raise ValueError("invalid opportunity status")

    def to_dict(self):
        value = super().to_dict()
        value["assessment"] = {key: item.to_dict() if isinstance(item, Assessment) else item
                                for key, item in self.assessment.items()}
        return value


class KnowledgePort(Protocol):
    version: str

    def search(
        self, *, case_id: str, source_hash: str, query: str, observations: Sequence[str], limit: int = 5
    ) -> list[KnowledgeHit]: ...
    def read(self, ref: str, *, max_bytes: int = 12000, offset: int = 0, paginate: bool = False) -> KnowledgeDocument: ...
    def contains(self, ref: str) -> bool: ...


class EvidencePort(Protocol):
    version: str
    available: bool

    def query(
        self, *, question: str, before: ProfileRef | None, after: ProfileRef | None, budget_bytes: int = 4096
    ) -> EvidenceBundle: ...


class ProgressPort(Protocol):
    version: str

    def view(self) -> ProgressView: ...
    def register_hypothesis(self, value: Hypothesis, *, source_ref=None, call_id=None) -> ProgressView: ...
    def apply_attempt(self, attempt: Attempt, evidence: Sequence[EvidenceBundle] | None, **kwargs) -> ProgressView: ...
    def add_evidence(self, cycle_id: str, bundle: EvidenceBundle, *, complete=False, call_id=None) -> ProgressView: ...
    def submit_step(self, value: dict, *, call_id=None) -> ProgressView: ...
    def select_program(self, program_ref: str, *, call_id=None) -> ProgressView: ...
