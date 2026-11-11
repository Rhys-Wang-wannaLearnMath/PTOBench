"""Finite schemas shared by MCP, Pi and in-process tests."""

from __future__ import annotations
import re
from ..foundation.protocol import ProtocolError


def obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


def string(n=1024, pattern=None):
    return {"type": "string", "minLength": 1, "maxLength": n, **({"pattern": pattern} if pattern else {})}


def array(n=16):
    return {"type": "array", "items": string(), "maxItems": n}


def integer(lo, hi):
    return {"type": "integer", "minimum": lo, "maximum": hi}


HASH = string(71, r"^(?:sha256:[0-9a-f]{64}|program:[a-z2-7]{16,32})$")
HYPOTHESIS = obj(
    {
        "parent_source_hash": HASH,
        "mechanism": string(128),
        "change_scope": array(),
        "preconditions": array(),
        # Up to eight questions can each require several registered signals.
        "expected_signals": array(64),
        "verification_questions": array(8),
        "rollback_condition": string(),
        "knowledge_refs": array(),
        "opportunity_id": string(128),
        "mechanism_id": string(128),
        "expected_revision": string(64, r'^revision:[a-z2-7]{16,32}$'),
    },
    (
        "parent_source_hash",
        "mechanism",
        "change_scope",
        "preconditions",
        "expected_signals",
        "verification_questions",
        "rollback_condition",
        "knowledge_refs",
        "opportunity_id",
        "expected_revision",
    ),
)
SPECS = [
    ("sluice_progress", "Read current loop progress, source identity, public budget and decision.", obj({})),
    ('sluice_profile_catalog', 'Read questions, expected signals and detail views for one profiling topic. No capture or hardware action.',
     obj({'topic': {'type': 'string', 'enum': ['timing', 'work', 'pmu', 'memory', 'compiler', 'dataflow']}}, ('topic',))),
    (
        "sluice_search",
        "Find relevant published knowledge candidates about source code mechanisms and constraints.",
        obj({"query": string(), "observations": array(), "limit": integer(1, 8),
             "cursor": string(2048), "topic": string(96), "browse": {"type": "boolean"}}, ()),
    ),
    (
        "sluice_read_skill",
        "Read complete knowledge and dependencies. Explicit paginate is allowed only for reference bodies; follow reference_page.next_offset.",
        obj({"ref": string(256), "max_bytes": integer(1, 32768), "offset": integer(0, 262144),
             "paginate": {"type": "boolean"}, "cursor": string(2048)}, ("ref",)),
    ),
    (
        "sluice_register",
        "Register the accepted optimize or recheck Decision as a testable hypothesis before editing Candidate Python source. Knowledge references may be empty for a justified new direction.",
        HYPOTHESIS,
    ),
    (
        'sluice_step',
        'Submit the Pi interpretation and the single next Decision. optimize/recheck/diagnose must cite a submitted or existing opportunity; diagnose also requires a registered evidence_question and does not create a Hypothesis or cycle. For a baseline diagnosis use sluice_profile_baseline, then reread progress and submit the next Decision. Assessment.refs may cite known Session evidence/profile/program refs or a published knowledge card returned by sluice_search; unrelated refs are rejected. Use knowledge_refs on sluice_register for the Hypothesis knowledge basis. For cycle references, the current opaque cycle:<ref> from context and the raw cycle_id from sluice_progress are both accepted.',
        obj({
            'expected_revision': string(64, r'^revision:[a-z2-7]{16,32}$'),
            'completed_cycle_id': string(128),
            'opportunities': {'type': 'array', 'maxItems': 32, 'items': obj({
                'opportunity_id': string(128), 'mechanism_id': string(128),
                'source_scope': array(32),
                'status': {'type': 'string', 'enum': ['open', 'needs_evidence', 'blocked', 'improved', 'recheck', 'closed']},
                'assessment': obj({key: obj({
                    'explanation': string(2000), 'refs': array(32), 'unknowns': array(32),
                }, ('explanation',)) for key in ('significance', 'evidence_strength', 'applicability',
                                                  'exploration', 'investigation_cost')}, ()),
                'cycle_refs': array(32), 'evidence_refs': array(32),
                'condition_change': string(1000),
            }, ('opportunity_id', 'source_scope', 'status'))},
            'decision': obj({
                'action': {'type': 'string', 'enum': ['optimize', 'diagnose', 'recheck', 'stop']},
                'opportunity_id': string(128), 'hypothesis_id': string(128),
                'evidence_question': string(128), 'reason': string(2000),
            }, ('action', 'reason')),
        }, ('expected_revision', 'decision')),
    ),
    (
        'sluice_select_program',
        'Restore a previously recorded verified program in this Session without a hardware action.',
        obj({'program_ref': string(64, r'^program:[a-z2-7]{16,32}$'),
             'expected_revision': string(64, r'^revision:[a-z2-7]{16,32}$')},
            ('program_ref', 'expected_revision')),
    ),
    (
        "sluice_evaluate",
        "Evaluate the registered candidate through the public PTOBench action.",
        obj({"source_hash": HASH}, ("source_hash",)),
    ),
    (
        'sluice_profile_baseline',
        'Inspect the current recorded source before registering a change. Reuses native profiling for the same recorded source and mode. Returns a baseline reference; never registers a hypothesis.',
        obj({'question': string()}, ('question',)),
    ),
    (
        "sluice_profile",
        "Answer a registered verification question after evaluate. Supply the parent baseline before_ref to compare a change. Questions sharing a native mode reuse one capture. Diagnostic decreases do not establish speedup.",
        obj(
            {
                "question": string(),
                "mode": {"type": "string", "enum": ["runtime", "pmu", "compiler", "clean"]},
                "before_ref": string(128),
                "after_ref": string(128),
            },
            ("question",),
        ),
    ),
    (
        'sluice_profile_detail',
        'Read a bounded evidence page without hardware work. See sluice_profile_catalog for view definitions. Follow next_cursor; use members for inputs/rank dispatches. In boundaries, task_ref filters one task. Hints contain ordered native-message fragments.',
        obj({'profile_ref':string(128),
             'view':{'type':'string','enum':['members','device_identity','execution_work','tasks','critical_path','engine_occupancy','dispatch_delays','scheduler_phases','boundaries','pmu','pmu_samples','resources','allocations','hints','samples','process_rounds','measured_host','pipeline','host']},
             'task_ref':string(128),'cursor':string(128),'budget_bytes':integer(1024,4096)},('profile_ref','view')),
    ),
    (
        'sluice_profile_raw',
        'List or read saved original diagnostic files by profile reference when runtime access is enabled. Read-only pages consume no hardware actions; omit file_ref to list, and follow next_offset. File bytes are base64 encoded.',
        obj({'profile_ref':string(128),'file_ref':string(128),'offset':integer(0,2**53-1)},('profile_ref',)),
    ),
    ("sluice_fork", "Fork the current registered workspace through the public PTOBench action.", obj({})),
    (
        "sluice_finish",
        "Request method completion with a concrete reason; PTOBench finalizes scores.",
        obj({"reason": string()}, ("reason",)),
    ),
]


def specifications():
    return [{"name": n, "description": d, "inputSchema": s} for n, d, s in SPECS]


def validate(value, schema):
    t = schema["type"]
    valid = {
        "object": type(value) is dict,
        "string": type(value) is str,
        "array": type(value) is list,
        "integer": type(value) is int,
        "boolean": type(value) is bool,
    }[t]
    if not valid:
        raise ProtocolError("Tool argument has an invalid JSON type")
    if "enum" in schema and value not in schema["enum"]:
        raise ProtocolError("Tool argument is outside the allowed values")
    if t == "object":
        if set(value) - schema["properties"].keys() or not set(schema["required"]) <= value.keys():
            raise ProtocolError("Tool arguments have unknown or missing fields")
        for k, v in value.items():
            validate(v, schema["properties"][k])
    if t == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 32768) or "\0" in value:
            raise ProtocolError("Tool text is outside the allowed bounds")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ProtocolError("Invalid source hash")
    if t == "array":
        if len(value) > schema["maxItems"]:
            raise ProtocolError("Tool list is too long")
        for item in value:
            validate(item, schema["items"])
    if t == "integer" and not schema["minimum"] <= value <= schema["maximum"]:
        raise ProtocolError("Tool integer is outside the allowed bounds")


def validate_call(name, args):
    for n, _, schema in SPECS:
        if n == name:
            validate(args, schema)
            return
    raise ProtocolError("Unknown SLUICE tool")
