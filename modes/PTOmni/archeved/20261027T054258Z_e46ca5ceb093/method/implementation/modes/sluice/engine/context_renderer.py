"""Build the bounded per-request projection from the trusted SLUICE controller."""

from __future__ import annotations

import json
from pathlib import Path

from .evidence_guard import opaque
from .public_output import project

MAX_CONTEXT_BYTES = 64 * 1024
_MESSAGE_ENVELOPE_BYTES = 128


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _contract(controller, first_round):
    name = "core_runtime.md" if first_round else "round_contract.md"
    path = Path(__file__).resolve().parents[1] / "resources" / "runtime_skills" / name
    if not path.is_file():
        raise ValueError("SLUICE round contract is unavailable")
    return path.read_text(encoding="utf-8")


def _public_hypothesis(value):
    if value is None:
        return None
    projected = value.to_dict()
    projected["hypothesis_id"] = opaque("hypothesis", value.hypothesis_id)
    result = project(projected)
    result["source_scope"] = list(value.change_scope)
    return result


def _public_decision(value):
    if value is None:
        return None
    projected = value.to_dict()
    if value.hypothesis_id:
        projected["hypothesis_id"] = opaque("hypothesis", value.hypothesis_id)
    return project(projected)


def _public_cycle(value):
    projected = value.to_dict()
    projected["cycle_id"] = opaque("cycle", value.cycle_id)
    if value.hypothesis_id:
        projected["hypothesis_id"] = opaque("hypothesis", value.hypothesis_id)
    return project(projected)


def _public_opportunity(value):
    projected = value.to_dict()
    projected["cycle_refs"] = [opaque("cycle", ref) for ref in value.cycle_refs]
    return project(projected)


def render(controller, *, max_bytes=MAX_CONTEXT_BYTES, first_round=False):
    if type(max_bytes) is not int or max_bytes < 1024:
        raise ValueError("context byte limit is invalid")
    if type(first_round) is not bool:
        raise ValueError("first_round must be a boolean")
    state = controller.progress.view()
    current = controller.source()
    working = next((item for item in state.programs if item.program_ref == state.working_program_ref), None)
    if working is None:
        raise ValueError("context_mismatch: working program is absent from the current state")
    dirty = current.content_hash != working.source_hash
    active = state.active_hypothesis
    pending = controller.pending
    if pending:
        phase = "diagnostics_pending"
    elif active:
        phase = "hypothesis_registered"
    elif state.decision:
        phase = "accepted_decision"
    elif state.cycles and state.cycles[-1].status in ("feedback_ready", "completed"):
        phase = "feedback_ready"
    else:
        phase = "initial_decision_needed"

    latest_cycle = state.cycles[-1] if state.cycles else None
    current_view = {
        "schema_version": 1,
        "state_revision": state.state_revision,
        "decision_phase": phase,
        "round_contract": _contract(controller, first_round=first_round),
        "contract_ref": "workspace:.benchmark_context/README.md",
        "program_view": {
            "working": state.working_program_ref,
            "last_verified": state.last_verified_program_ref,
            "best_verified": state.best_verified_program_ref,
            "current_source": opaque("program", current.content_hash),
            "dirty": dirty,
            "candidate_python_files": sorted(item["path"] for item in current.files
                                              if item["path"].endswith(".py")),
        },
        "active_hypothesis": _public_hypothesis(active),
        "accepted_decision": _public_decision(state.decision),
        "opportunities": [_public_opportunity(item) for item in state.opportunities],
        "recent_cycle": _public_cycle(latest_cycle) if latest_cycle else None,
        "relevant_history": [_public_cycle(item) for item in list(state.cycles[:-1])[-12:]],
        "evidence": [item.to_dict() for item in state.evidence[-12:]],
        "unknowns": sorted({unknown for cycle in state.cycles[-12:] for unknown in cycle.unknowns}),
        "remaining_hardware_actions": controller.remaining if controller.remaining is not None else "unlimited",
        "pending_questions": list(pending.get("questions", ())) if pending else [],
        "omissions": [],
    }

    refs = []
    if active:
        refs.extend(active.knowledge_refs)
    refs.extend(item.mechanism_id if str(item.mechanism_id).startswith(("card:", "ref:"))
                else "card:" + str(item.mechanism_id)
                for item in state.opportunities if item.mechanism_id)
    refs = list(dict.fromkeys(refs))
    excerpts = []
    for ref in refs[:8]:
        try:
            document = controller.knowledge.read(ref, max_bytes=4096)
        except Exception:
            current_view["omissions"].append({"ref": ref, "reason": "knowledge_unavailable"})
            continue
        if document.needs_more_budget or document.truncated:
            current_view["omissions"].append({"ref": ref, "reason": "read_full_reference_on_demand"})
            continue
        excerpts.append({"ref": ref, "content": document.content})
    current_view["knowledge_refs_and_excerpts"] = excerpts

    # Keep every required current fact intact; drop oldest history and then
    # optional excerpts if an unusually large Session exceeds the brief limit.
    payload_limit = max_bytes - _MESSAGE_ENVELOPE_BYTES
    while len(_encode(current_view)) > payload_limit and current_view["relevant_history"]:
        removed = current_view["relevant_history"].pop(0)
        current_view["omissions"].append({"cycle_id": removed["cycle_id"], "reason": "context_budget_old_history"})
    while len(_encode(current_view)) > payload_limit and current_view["knowledge_refs_and_excerpts"]:
        removed = current_view["knowledge_refs_and_excerpts"].pop()
        current_view["omissions"].append({"ref": removed["ref"], "reason": "context_budget_knowledge_excerpt"})
    if len(_encode(current_view)) > payload_limit:
        raise ValueError("context_too_large: required current state exceeds 64 KiB")
    current_view = project(current_view)
    encoded = _encode(current_view)
    if len(encoded) > payload_limit:
        raise ValueError("context_too_large: public projection exceeds 64 KiB")
    return current_view
