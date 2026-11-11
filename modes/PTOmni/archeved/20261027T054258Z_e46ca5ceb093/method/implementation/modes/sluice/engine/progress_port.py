"""The single SLUICE state reducer and its file/in-memory journal adapters."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import os

from .contracts import (
    Assessment, Attempt, Decision, EvidenceBundle, Hypothesis, OptimizationCycle,
    Opportunity, ProgramVersion, ProgressView, digest,
)
from .evidence_guard import opaque
from ..foundation.protocol import ProtocolError


def project(events):
    """Replay v1 event shapes and current v2 events through one deterministic reducer."""
    active = None
    attempts, evidence, programs, cycles = [], [], {}, []
    opportunities = {}
    decision = None
    working = last_verified = best_verified = None
    best_attempt = None
    pending_cycle = None
    seen_attempts = set()
    cycle_hypotheses = {}

    def ensure_program(source_hash, *, case_id, parent=None, source_ref=None, action_id=None,
                       snapshot_id=None, correctness=None, latency=None, failed=False):
        nonlocal working, last_verified, best_verified, best_attempt
        ref = opaque('program', source_hash)
        if ref not in programs:
            parent_ref = opaque('program', parent) if parent else None
            programs[ref] = ProgramVersion(
                ref, case_id, source_hash, parent_ref, source_ref, action_id, snapshot_id,
                'unverified', None, (parent_ref,) if parent_ref else (),
            )
        working = ref
        if correctness == 'PASS' and latency is not None and latency > 0:
            last_verified = ref
            previous_latency = programs[ref].latency_us
            candidate_latency = min(latency, previous_latency) if previous_latency is not None else latency
            improved_program_sample = previous_latency is None or latency < previous_latency
            programs[ref] = replace(programs[ref], verification='verified', latency_us=candidate_latency,
                                    action_id=(action_id if improved_program_sample else programs[ref].action_id),
                                    snapshot_id=(snapshot_id if improved_program_sample else programs[ref].snapshot_id))
            if best_attempt is None or candidate_latency < best_attempt.latency_us:
                if best_verified and best_verified in programs:
                    programs[best_verified] = replace(programs[best_verified], verification='verified')
                best_verified = ref
                programs[ref] = replace(programs[ref], verification='best')
                best_attempt = next((item for item in reversed(attempts)
                                     if opaque('program', item.child_source_hash) == ref
                                     and item.correctness == 'PASS'
                                     and item.latency_us == latency), best_attempt)
            elif ref == best_verified:
                programs[ref] = replace(programs[ref], verification='best')
        elif correctness == 'FAIL':
            programs[ref] = replace(programs[ref], verification='failed', latency_us=None,
                                    action_id=action_id or programs[ref].action_id)
        elif failed:
            programs[ref] = replace(programs[ref], verification='failed', latency_us=None,
                                    action_id=action_id or programs[ref].action_id)
        return ref

    for index, event in enumerate(events):
        kind = event.get('kind')
        if kind == 'registration':
            if active is not None:
                raise ProtocolError('A hypothesis is already active')
            active = Hypothesis.from_dict(event['value'])
            if decision and decision.action in ('optimize', 'recheck') and decision.opportunity_id == active.opportunity_id:
                decision = replace(decision, hypothesis_id=active.hypothesis_id)
            parent = active.parent_source_hash
            parent_ref = opaque('program', parent)
            if parent_ref not in programs:
                programs[parent_ref] = ProgramVersion(
                    parent_ref, active.case_id, parent, source_ref=event.get('source_ref'),
                    verification='seed',
                )
                if last_verified is None:
                    last_verified = parent_ref
                if best_verified is None:
                    best_verified = parent_ref
            if working is None:
                working = parent_ref
            cycle_id = 'cycle-' + active.hypothesis_id
            pending_cycle = {
                'cycle_id': cycle_id, 'kind': active.kind, 'program_ref': parent_ref,
                'parent_program_ref': parent_ref, 'hypothesis_id': active.hypothesis_id,
                'action_id': None, 'result': 'pending', 'evidence_refs': [],
                'status': 'pending', 'unknowns': [],
            }
            cycle_hypotheses[cycle_id] = opaque('hypothesis', active.hypothesis_id)
        elif kind == 'seed':
            value = event['value']
            ref = opaque('program', value['source_hash'])
            if ref not in programs:
                programs[ref] = ProgramVersion(
                    ref, value['case_id'], value['source_hash'], source_ref=value['source_ref'],
                    verification='seed',
                )
            if working is None:
                working = ref
            if last_verified is None:
                last_verified = ref
            if best_verified is None:
                best_verified = ref
        elif kind == 'attempt':
            attempt = Attempt.from_dict(event['value'])
            if not active or (attempt.hypothesis_id, attempt.case_id, attempt.parent_source_hash) != (
                active.hypothesis_id, active.case_id, active.parent_source_hash,
            ):
                raise ProtocolError('Attempt does not match the active hypothesis')
            if attempt.attempt_id in seen_attempts:
                raise ProtocolError('Duplicate attempt')
            if not attempt.action_id or not attempt.observation_ref or not attempt.child_source_hash:
                raise ProtocolError('Attempt provenance is missing')
            seen_attempts.add(attempt.attempt_id)
            attempts.append(attempt)
            raw_bundles = event.get('evidence')
            bundles = []
            if raw_bundles is not None:
                if isinstance(raw_bundles, dict):  # read-only compatibility with v1 journals
                    raw_bundles = [raw_bundles]
                bundles = [EvidenceBundle.from_dict(item) for item in raw_bundles]
                if len({bundle.question for bundle in bundles}) != len(bundles):
                    raise ProtocolError('Evidence questions must be distinct bundles')
                evidence.extend(bundles)
                available = sum(bundle.status == 'available' for bundle in bundles)
                attempt = replace(attempt, evidence_status=(
                    'available' if bundles and available == len(bundles)
                    else 'partial' if available
                    else 'unavailable' if bundles else attempt.evidence_status
                ))
                attempts[-1] = attempt
            prior_ref = opaque('program', attempt.parent_source_hash)
            source_ref = event.get('source_ref')
            snapshot_id = event.get('snapshot_id')
            child_ref = ensure_program(
                attempt.child_source_hash, case_id=attempt.case_id, parent=attempt.parent_source_hash,
                source_ref=source_ref, action_id=attempt.action_id, snapshot_id=snapshot_id,
                correctness=attempt.correctness if attempt.outcome not in ('source_mismatch', 'action_failed') else None,
                latency=attempt.latency_us if attempt.correctness == 'PASS' else None,
                failed=attempt.outcome in ('correctness_fail', 'compile_fail'),
            )
            if pending_cycle:
                pending_cycle.update(program_ref=child_ref, parent_program_ref=prior_ref,
                                     action_id=attempt.action_id, result=attempt.outcome,
                                         evidence_refs=[bundle.evidence_ref for bundle in bundles],
                                     status=('collecting_evidence' if event.get('evidence_pending')
                                             else 'feedback_ready'),
                                     unknowns=[missing for bundle in bundles for missing in bundle.missing])
                cycles.append(OptimizationCycle(**pending_cycle))
                cycle_hypotheses[pending_cycle['cycle_id']] = opaque('hypothesis', attempt.hypothesis_id)
                pending_cycle = None
            best_attempt = next((item for item in reversed(attempts)
                                 if best_verified == opaque('program', item.child_source_hash)
                                 and item.correctness == 'PASS'), best_attempt)
            active = None
            # A Decision belongs to the state revision on which Pi proposed it.
            # A completed attempt changes the state and requires a fresh interpretation.
            decision = None
        elif kind == 'evidence':
            cycle_id = event['cycle_id']
            cycle_positions = [i for i, cycle in enumerate(cycles) if cycle.cycle_id == cycle_id]
            if len(cycle_positions) != 1:
                raise ProtocolError('Evidence does not reference a known cycle')
            position = cycle_positions[0]
            cycle = cycles[position]
            bundle = EvidenceBundle.from_dict(event['value'])
            if cycle.status != 'collecting_evidence':
                raise ProtocolError('Cycle is not collecting evidence')
            if bundle.hypothesis_ref != cycle_hypotheses.get(cycle_id) or bundle.program_ref != cycle.program_ref:
                raise ProtocolError('Evidence identity does not match its cycle')
            if any(item.evidence_ref == bundle.evidence_ref for item in evidence):
                raise ProtocolError('Duplicate evidence reference')
            if any(item.question == bundle.question for item in evidence
                   if item.hypothesis_ref == bundle.hypothesis_ref and item.program_ref == bundle.program_ref):
                raise ProtocolError('Question evidence already exists for this program')
            evidence.append(bundle)
            refs = (*cycle.evidence_refs, bundle.evidence_ref)
            unknowns = (*cycle.unknowns, *bundle.missing)
            status = 'feedback_ready' if event.get('complete') else cycle.status
            cycles[position] = replace(cycle, evidence_refs=refs, unknowns=unknowns, status=status)
            matching = next((i for i, item in enumerate(attempts)
                             if item.hypothesis_id == bundle.hypothesis_ref.removeprefix('hypothesis:')
                             or opaque('hypothesis', item.hypothesis_id) == bundle.hypothesis_ref), None)
            if matching is not None:
                siblings = [item for item in evidence
                            if item.hypothesis_ref == bundle.hypothesis_ref
                            and item.program_ref == bundle.program_ref]
                available_count = sum(item.status == 'available' for item in siblings)
                attempts[matching] = replace(attempts[matching], evidence_status=(
                    'available' if available_count == len(siblings)
                    else 'partial' if available_count
                    else 'unavailable'
                ))
        elif kind == 'step':
            value = event['value']
            proposed = Decision(**{k: v for k, v in value['decision'].items() if k != 'schema_version'})
            completed_cycle_id = value.get('completed_cycle_id')
            if completed_cycle_id:
                matches = [i for i, cycle in enumerate(cycles) if cycle.cycle_id == completed_cycle_id]
                if len(matches) != 1 or cycles[matches[0]].status not in ('feedback_ready', 'completed'):
                    raise ProtocolError('Step does not reference a feedback-ready cycle')
                position = matches[0]
                cycles[position] = replace(
                    cycles[position],
                    status='terminal' if proposed.action == 'stop' else 'completed',
                )
            for raw in value.get('opportunities', []):
                assessment = {
                    key: item if isinstance(item, Assessment) else Assessment(**item)
                    for key, item in raw.get('assessment', {}).items()
                }
                opportunities[raw['opportunity_id']] = Opportunity(
                    opportunity_id=raw['opportunity_id'],
                    mechanism_id=raw.get('mechanism_id'),
                    source_scope=tuple(raw.get('source_scope', ())),
                    status=raw.get('status', 'open'), assessment=assessment,
                    cycle_refs=tuple(raw.get('cycle_refs', ())),
                    evidence_refs=tuple(raw.get('evidence_refs', ())),
                    condition_change=raw.get('condition_change'),
                )
            decision = proposed
        elif kind == 'program_selected':
            ref = event['program_ref']
            if ref not in programs:
                raise ProtocolError('Selected program is not in this Session')
            working = ref
            decision = None
        elif kind == 'decision':  # legacy journal event; retain only as an old projection
            raw = event.get('value')
            if raw:
                decision = Decision(**{k: v for k, v in raw.items() if k != 'schema_version'})
        else:
            raise ProtocolError('Unsupported progress event')

    best_program = programs.get(best_verified)
    best = next((a for a in reversed(attempts)
                 if best_verified == opaque('program', a.child_source_hash)
                 and a.correctness == 'PASS' and best_program is not None
                 and a.latency_us == best_program.latency_us), None)
    if best is None and best_verified and best_verified in programs and programs[best_verified].verification == 'seed':
        best = None
    revision = digest(events)
    return ProgressView(
        revision, active, tuple(attempts), best, tuple(evidence), False,
        tuple(programs.values()), working, last_verified, best_verified,
        tuple(cycles), tuple(opportunities.values()), decision,
        opaque('revision', revision),
    )


class _ProgressPortBase:
    """All storage adapters delegate state changes and projection to this reducer."""
    def _events(self):
        raise NotImplementedError

    def _commit(self, event):
        raise NotImplementedError

    def view(self):
        return project(self._events())

    def _append(self, event):
        events = self._events()
        call_id = event.get('call_id')
        if call_id:
            existing = next((item for item in events if item.get('call_id') == call_id), None)
            if existing:
                if existing != event:
                    raise ProtocolError('Idempotency key was reused with different content')
                return project(events)
        result = project([*events, event])
        self._commit(event)
        return result

    def register_hypothesis(self, value, *, source_ref=None, call_id=None):
        return self._append({'kind': 'registration', 'value': value.to_dict(),
                             'source_ref': source_ref, 'call_id': call_id})

    def initialize_program(self, *, case_id, source_hash, source_ref, call_id='initial-seed'):
        return self._append({'kind': 'seed', 'value': {
            'case_id': case_id, 'source_hash': source_hash, 'source_ref': source_ref,
        }, 'call_id': call_id})

    def apply_attempt(self, attempt, evidence=None, *, source_ref=None, snapshot_id=None,
                      evidence_pending=False, call_id=None):
        bundles = evidence if isinstance(evidence, (tuple, list)) else ([evidence] if evidence else [])
        return self._append({
            'kind': 'attempt', 'value': attempt.to_dict(),
            'evidence': [bundle.to_dict() for bundle in bundles],
            'source_ref': source_ref, 'snapshot_id': snapshot_id, 'call_id': call_id,
            'evidence_pending': evidence_pending,
        })

    def add_evidence(self, cycle_id, bundle, *, complete=False, call_id=None):
        return self._append({'kind': 'evidence', 'cycle_id': cycle_id,
                             'value': bundle.to_dict(), 'complete': complete,
                             'call_id': call_id})

    def submit_step(self, value, *, call_id=None):
        return self._append({'kind': 'step', 'value': value, 'call_id': call_id})

    def select_program(self, program_ref, *, call_id=None):
        return self._append({'kind': 'program_selected', 'program_ref': program_ref, 'call_id': call_id})


class JournalProgressPort(_ProgressPortBase):
    """Production append-only journal; v1 events replay as read-only history."""
    version = 'sluice-state-journal-v2'

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _events(self):
        if self.path.is_symlink():
            raise ProtocolError('Progress journal cannot be a symlink')
        return [json.loads(s) for s in self.path.read_text().splitlines() if s] if self.path.exists() else []

    def _commit(self, event):
        with self.path.open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


class JsonProgressPort(_ProgressPortBase):
    """In-memory test adapter using the production reducer unchanged."""
    version = 'sluice-test-state-v2'

    def __init__(self, events=None):
        self.events = json.loads(json.dumps(events or []))

    def _events(self):
        return self.events

    def _commit(self, value):
        self.events.append(value)
