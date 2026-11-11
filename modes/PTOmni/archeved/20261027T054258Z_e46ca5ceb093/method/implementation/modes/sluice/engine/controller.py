"""One active hypothesis; native tools drive the hypothesis/change/verify/update loop."""

from __future__ import annotations
from dataclasses import fields, replace
import json
import math
import hashlib
from pathlib import Path
import shutil
import stat
import threading
import uuid

from .contracts import Attempt, Decision, EvidenceBundle, Hypothesis, ProfileRef
from .evidence_guard import opaque
from .evidence_port import UnavailableEvidencePort
from .knowledge_port import LocalSkillKnowledgePort, bound_knowledge
from .progress_port import JournalProgressPort
from .tools import validate_call
from ..foundation.actions import ActionObservation, PublicActionRunner
from ..foundation.events import EventLedger
from ..foundation.protocol import ProtocolError
from ..foundation.source import fingerprint


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ProtocolError("Record path cannot be a symlink")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.is_symlink():
        raise ProtocolError("Record temporary path cannot be a symlink")
    tmp.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def file_hash(path):
    return 'sha256:' + hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Controller:
    def __init__(
        self,
        context,
        *,
        knowledge=None,
        evidence=None,
        progress=None,
        actions=None,
        resources=None,
        cancel=None,
    ):
        self.context = context
        self.ledger = EventLedger(context)
        if knowledge is not None:
            self.knowledge = knowledge
        else:
            published = context.method_root / "resources/knowledge"
            if (published / "index/search.json").is_file():
                self.knowledge = bound_knowledge(published)
            elif published.exists():
                # A present but incomplete production directory is a hard error; never hide a
                # broken package behind the source-backed compatibility port.
                raise ProtocolError("Knowledge package is unavailable")
            elif resources is not None:
                # Development fallback is explicit at construction time.  A missing production
                # package must never silently expose the legacy source-backed skill port.
                self.knowledge = LocalSkillKnowledgePort(resources)
            else:
                raise ProtocolError("Knowledge package is unavailable")
        self.evidence = evidence or UnavailableEvidencePort()
        self.progress = progress or JournalProgressPort(context.work_root / "progress.jsonl")
        self.actions = actions or PublicActionRunner(context, self.ledger)
        self.cancel = cancel or threading.Event()
        self.lock = threading.RLock()
        self.closed = context.work_root / "closed.json"
        self.pending_path = context.work_root / "pending.json"
        self.source_path = context.work_root / "registered_source.json"
        self.intent_path = context.work_root / "action_intent.json"
        self.pending = json.loads(self.pending_path.read_text()) if self.pending_path.exists() else None
        self.remaining = context.budget["max_hardware_actions"]
        if self.remaining == "unlimited":
            self.remaining = None
        for event in self.ledger.events():
            if event["event_type"] in ("action_finished", "action_failed") and event.get("payload_ref"):
                p = context.session_root / event["payload_ref"]
                if p.is_file():
                    r = json.loads(p.read_text()).get("cli_result") or {}
                    if type(r.get("remaining_hardware_actions")) is int:
                        self.remaining = (
                            r["remaining_hardware_actions"]
                            if self.remaining is None
                            else min(self.remaining, r["remaining_hardware_actions"])
                        )
        config_path = context.method_root / "config.json"
        prepared = json.loads(config_path.read_text()) if config_path.is_file() else {}
        state = self.progress.view()
        if not state.programs:
            source = self.source()
            source_ref = self._archive_source(source.content_hash)
            self.progress.initialize_program(case_id=context.case_id,
                                             source_hash=source.content_hash,
                                             source_ref=source_ref)
            state = self.progress.view()
        working = next((p.source_hash for p in state.programs
                        if p.program_ref == state.working_program_ref), None)
        if state.attempts and state.attempts[-1].correctness != 'PASS':
            working = next((p.source_hash for p in state.programs
                            if p.program_ref == state.last_verified_program_ref), None)
        self.accepted_source_hash = working or prepared.get(
            "source", {}).get("content_sha256", self.source().content_hash
        )
        self.refs_path = context.work_root / 'profile_refs.json'
        self.refs = {}
        if self.refs_path.is_file():
            self.refs = {k: ProfileRef(**v) for k, v in json.loads(self.refs_path.read_text()).items()}
        self.public_evidence_refs_path = context.work_root / 'public_evidence_refs.json'
        self.public_evidence_refs = {}
        if self.public_evidence_refs_path.is_file():
            self.public_evidence_refs = json.loads(self.public_evidence_refs_path.read_text())
        self.recovered_profiles = {}
        self._recover()

    def _recover(self):
        if self.pending:
            committed = next(
                (
                    a
                    for a in self.progress.view().attempts
                    if a.attempt_id == self.pending["attempt"]["attempt_id"]
                ),
                None,
            )
            if committed is not None:
                state = self.progress.view()
                known = {bundle.evidence_ref for bundle in state.evidence}
                known_questions = {bundle.question for bundle in state.evidence
                                   if bundle.hypothesis_ref == opaque('hypothesis', self.pending.get('hypothesis_id', ''))
                                   and bundle.program_ref == opaque('program', self.pending['attempt']['child_source_hash'])}
                cycle = next((item for item in reversed(state.cycles)
                              if item.hypothesis_id == self.pending.get('hypothesis_id')), None)
                if cycle:
                    for raw_bundle in self.pending.get('evidence', []):
                        bundle = EvidenceBundle.from_dict(raw_bundle)
                        if bundle.evidence_ref not in known:
                            complete = not self.pending['questions']
                            self.progress.add_evidence(
                                cycle.cycle_id, bundle, complete=complete,
                                call_id='evidence:' + cycle.cycle_id + ':' + opaque('question', bundle.question),
                            )
                            known.add(bundle.evidence_ref)
                            known_questions.add(bundle.question)
                self.pending['questions'] = [q for q in self.pending['questions'] if q not in known_questions]
                self._archive_attempt(committed)
                if not self.pending['questions']:
                    self.pending_path.unlink(missing_ok=True)
                    self.pending = None
                else:
                    save(self.pending_path, self.pending)
            else:
                self._apply()
        if not self.intent_path.exists():
            return
        intent = json.loads(self.intent_path.read_text())
        action_id = intent.get("method_action_id")
        if not action_id:
            return
        try:
            uuid.UUID(action_id)
        except (ValueError, TypeError):
            return
        path = self.context.work_root / "actions" / action_id / "result.json"
        if not path.is_file():
            return  # Existing live/incomplete handle is never restarted.
        raw = json.loads(path.read_text())
        obs = ActionObservation(**{f.name: raw[f.name] for f in fields(ActionObservation)})
        h = self.progress.view().active_hypothesis or self._pending_hypothesis()
        if obs.method_action_id != action_id or obs.operation != intent["operation"]:
            return
        if obs.profile_mode != intent.get('mode'):
            return
        self._observe_budget(obs)
        if intent.get('purpose') == 'baseline':
            if (intent.get('source_hash') != obs.source_before or obs.source_after != obs.source_before
                    or intent.get('mode') != obs.profile_mode):
                return
            ref = self._baseline_ref(obs)
            self.intent_path.unlink()
            return
        if obs.operation == 'evaluate' and self.pending and self.pending['attempt'].get('action_id') == action_id:
            self.intent_path.unlink()
            return
        if obs.operation == 'profile' and self.pending is None and any(
            ref.action_id == action_id for ref in self.refs.values()
        ):
            self.intent_path.unlink()
            return
        if any(a.action_id == action_id for a in self.progress.view().attempts):
            self.intent_path.unlink()
            return
        if obs.operation == "evaluate":
            if not h or h.hypothesis_id != intent.get("hypothesis_id"):
                return
            if self.pending is None:
                self._complete_evaluate(h, obs, intent["source_hash"])
        elif obs.operation == "profile":
            if self.pending is None and any(
                a.hypothesis_id == intent.get("hypothesis_id")
                and a.child_source_hash == intent.get("source_hash") == obs.source_before
                for a in self.progress.view().attempts
            ):
                self.intent_path.unlink()
                return
            if not self.pending or not h or h.hypothesis_id != intent.get("hypothesis_id"):
                return
            ref = self._profile_ref(obs)
            self.refs[ref.ref] = ref
            self._save_refs()
            if intent.get('question') in self.pending['questions']:
                self.pending.setdefault('profile_actions', {})[intent['question']] = ref.ref
                save(self.pending_path, self.pending)
                self.recovered_profiles[intent['question']] = ref
        self.intent_path.unlink()

    def _pending_hypothesis(self):
        if not self.pending:
            return None
        hypothesis_id = self.pending.get('hypothesis_id') or self.pending.get('attempt', {}).get('hypothesis_id')
        path = self.context.work_root / 'hypotheses' / f'{hypothesis_id}.json'
        if not hypothesis_id or not path.is_file():
            return None
        return Hypothesis.from_dict(json.loads(path.read_text()))

    def source(self):
        current = fingerprint(self.context.workspace)
        self.validate_source(current)
        return current

    def validate_source(self, current):
        """Only Python source may change; preserve all initial configuration."""
        path = self.context.work_root / 'fixed_source_files.json'
        fixed = [f for f in current.files if not f['path'].endswith('.py')]
        if path.is_file():
            expected = json.loads(path.read_text())
        else:
            config = self.context.method_root / 'config.json'
            prepared = json.loads(config.read_text()) if config.is_file() else {}
            original = prepared.get('source', {}).get('files', current.files)
            expected = [f for f in original if not f['path'].endswith('.py')]
            save(path, expected)
        if fixed != expected:
            raise ProtocolError('Only Candidate Python source may change; configuration is fixed')

    def _archive_source(self, expected_hash=None):
        """Save one exact workspace version before another action can overwrite it."""
        current = self.source()
        if expected_hash is not None and current.content_hash != expected_hash:
            raise ProtocolError('Source changed before program version archival')
        ref = opaque('program', current.content_hash)
        key = ref.split(':', 1)[1]
        programs = self.context.work_root / 'programs'
        target = programs / key
        if target.exists():
            if fingerprint(target).content_hash != current.content_hash:
                raise ProtocolError('Program version archive identity mismatch')
            return str(target.relative_to(self.context.session_root))
        staging = programs / ('.' + key + '.incoming')
        if staging.exists() or staging.is_symlink():
            raise ProtocolError('Program archive staging path already exists')
        staging.mkdir(parents=True)
        try:
            for directory in current.directories:
                if directory == '.':
                    continue
                (staging / directory).mkdir(parents=True, exist_ok=True)
            for item in current.files:
                source = self.context.workspace / item['path']
                if source.is_symlink() or not source.is_file():
                    raise ProtocolError('Program source changed during archival')
                destination = staging / item['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if file_hash(destination) != item['sha256']:
                    raise ProtocolError('Program source changed during archival')
            if fingerprint(staging).content_hash != current.content_hash:
                raise ProtocolError('Archived program identity mismatch')
            for path in sorted(staging.rglob('*'), reverse=True):
                if path.is_file():
                    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                else:
                    path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            staging.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            staging.replace(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return str(target.relative_to(self.context.session_root))

    def _select_program(self, args, *, call_id=None):
        self._open()
        state = self.progress.view()
        if args['expected_revision'] != state.state_revision:
            raise ProtocolError('Stale state revision; reread progress before selecting a program')
        if state.active_hypothesis or self.pending or self.intent_path.exists():
            raise ProtocolError('Program selection requires a completed cycle and no active action')
        program = next((p for p in state.programs if p.program_ref == args['program_ref']), None)
        if program is None:
            raise ProtocolError('Program reference is not registered in this Session')
        if program.verification not in ('verified', 'best', 'seed') or not program.source_ref:
            raise ProtocolError('Only archived seed or correctness-verified programs can be selected')
        archive = self.context.session_root / program.source_ref
        if archive.is_symlink() or not archive.is_dir() or fingerprint(archive).content_hash != program.source_hash:
            raise ProtocolError('Program archive is missing or damaged')
        current = self.source()
        self.validate_source(current)
        if state.working_program_ref == program.program_ref and current.content_hash == program.source_hash:
            return {'program_ref': program.program_ref, 'source_hash': opaque('program', current.content_hash),
                    'state_revision': state.state_revision, 'hardware_actions_used': 0}
        archive_files = {item['path']: item for item in fingerprint(archive).files}
        current_python = {item['path'] for item in current.files if item['path'].endswith('.py')}
        archived_python = {name for name in archive_files if name.endswith('.py')}
        fixed = {item['path']: item['sha256'] for item in current.files if not item['path'].endswith('.py')}
        archived_fixed = {name: item['sha256'] for name, item in archive_files.items() if not name.endswith('.py')}
        if fixed != archived_fixed:
            raise ProtocolError('Program archive does not match fixed Case files')
        for name in current_python - archived_python:
            path = self.context.workspace / name
            if path.is_symlink() or not path.is_file():
                raise ProtocolError('Current source changed during program selection')
            path.unlink()
        for name in archived_python:
            source, target = archive / name, self.context.workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.sluice-restore')
            if temporary.exists() or temporary.is_symlink():
                raise ProtocolError('Program restore path already exists')
            shutil.copyfile(source, temporary)
            if file_hash(temporary) != archive_files[name]['sha256']:
                temporary.unlink(missing_ok=True)
                raise ProtocolError('Program archive changed during selection')
            temporary.replace(target)
        restored = self.source()
        if restored.content_hash != program.source_hash:
            raise ProtocolError('Selected program source identity mismatch')
        self.progress.select_program(program.program_ref, call_id=call_id)
        self.accepted_source_hash = restored.content_hash
        return {'program_ref': program.program_ref, 'source_hash': opaque('program', restored.content_hash),
                'state_revision': self.progress.view().state_revision, 'hardware_actions_used': 0}

    def emit(self, kind, ref, **payload):
        return self.ledger.append(
            kind, source_hash=self.source().content_hash, payload_ref=ref, payload=payload
        )

    def decision(self):
        h = self.progress.view().active_hypothesis or self._pending_hypothesis()
        if self.closed.exists() or self.cancel.is_set():
            return Decision("stop", "Method execution is closed", h.hypothesis_id if h else None)
        if self.intent_path.exists():
            return Decision("stop", "An interrupted public action needs reconciliation",
                            h.hypothesis_id if h else None)
        if self.pending:
            question = self.pending['questions'][0] if self.pending.get('questions') else None
            return Decision("diagnose", "Answer the remaining registered verification questions",
                            h.hypothesis_id if h else None, evidence_question=question)
        if self.remaining is not None and self.remaining <= 0:
            return Decision("stop", "Public hardware budget is exhausted")
        return self.progress.view().decision

    def view(self):
        progress = self.progress.view().to_dict()
        progress["attempt_count"] = len(progress["attempts"])
        progress["attempts"] = progress["attempts"][-8:]
        progress["evidence"] = progress["evidence"][-3:]
        return {
            "progress": progress,
            "source_hash": self.source().content_hash,
            "case": self.context.public_dict()["case"],
            "remaining_hardware_actions": self.remaining if self.remaining is not None else "unlimited",
            "decision": self.decision().to_dict() if self.decision() else None,
            "pending_questions": self.pending["questions"] if self.pending else [],
            "profile_questions": self.evidence.capabilities() if hasattr(self.evidence, 'capabilities') else [],
            "profile_acquisition": self._profile_plan(),
            "available_profiles": self._available_profiles(),
            "profile_reference_count": len(self.refs),
            "profiles_truncated": len(self.refs) > 8,
        }

    def context_projection(self):
        from .context_renderer import render, _encode
        directory = self.context.work_root / 'context_rounds'
        with self.lock:
            directory.mkdir(parents=True, exist_ok=True)
            sequence = len(list(directory.glob('round-*.json'))) + 1
            round_id = 'round-%06d' % sequence
            first_round = not any(
                json.loads(path.read_text()).get('sent_view') is not None
                for path in directory.glob('round-*.json')
            )
            projection = render(self, first_round=first_round)
            save(directory / (round_id + '.json'), {
                'schema_version': 1,
                'round_id': round_id,
                'state_revision': projection['state_revision'],
                'program_ref': projection['program_view']['working'],
                'cycle_ref': (projection['recent_cycle'] or {}).get('cycle_id'),
                'round_context_bytes': len(_encode(projection)),
                'round_context': projection,
                'sent_view': None,
            })
            return {'round_id': round_id, 'projection': projection}

    def record_context_sent(self, args):
        from .context_renderer import _encode
        if set(args) != {'round_id', 'state_revision', 'message', 'message_manifest'}:
            raise ProtocolError('Invalid context sent record')
        if (not isinstance(args['round_id'], str) or len(args['round_id']) != 12 or
                not args['round_id'].startswith('round-') or not args['round_id'][6:].isdigit()):
            raise ProtocolError('Invalid context round reference')
        if not isinstance(args['state_revision'], str):
            raise ProtocolError('Invalid context state revision')
        if not isinstance(args['message'], str) or not isinstance(args['message_manifest'], dict):
            raise ProtocolError('Invalid context sent view')
        path = self.context.work_root / 'context_rounds' / (args['round_id'] + '.json')
        with self.lock:
            if not path.is_file() or path.is_symlink():
                raise ProtocolError('Unknown context round reference')
            record = json.loads(path.read_text())
            if record['state_revision'] != args['state_revision']:
                raise ProtocolError('Context revision does not match its projection')
            if record.get('sent_view') is not None:
                raise ProtocolError('Context round was already recorded')
            if self.progress.view().state_revision != args['state_revision']:
                raise ProtocolError('Context state changed before request dispatch')
            from .evidence_guard import opaque
            current_source_ref = opaque('program', self.source().content_hash)
            if current_source_ref != record['round_context']['program_view']['current_source']:
                raise ProtocolError('Program source changed before request dispatch')
            manifest = args['message_manifest']
            if set(manifest) != {'message_count', 'conversation_sha256', 'tool_call_ids', 'tool_result_ids'}:
                raise ProtocolError('Context message manifest has unknown fields')
            if (type(manifest['message_count']) is not int or not 1 <= manifest['message_count'] <= 100000 or
                    not isinstance(manifest['conversation_sha256'], str) or
                    len(manifest['conversation_sha256']) != 64 or
                    any(c not in '0123456789abcdef' for c in manifest['conversation_sha256']) or
                    not isinstance(manifest['tool_call_ids'], list) or
                    not isinstance(manifest['tool_result_ids'], list)):
                raise ProtocolError('Context message manifest is invalid')
            calls, results = manifest['tool_call_ids'], manifest['tool_result_ids']
            if (len(calls) > 4096 or len(results) > 4096 or
                    any(not isinstance(value, str) or not value or len(value) > 256 for value in calls + results) or
                    sorted(calls) != sorted(results)):
                raise ProtocolError('Context tool call/result manifest is incomplete')
            prefix = '<!-- SLUICE_ROUND_CONTEXT: ' + args['round_id'] + ' -->\n'
            suffix = '\n<!-- /SLUICE_ROUND_CONTEXT -->'
            if not args['message'].startswith(prefix) or not args['message'].endswith(suffix):
                raise ProtocolError('Context sent view has invalid framing')
            try:
                sent_projection = json.loads(args['message'][len(prefix):-len(suffix)])
            except json.JSONDecodeError as exc:
                raise ProtocolError('Context sent view is invalid JSON') from exc
            if sent_projection != record['round_context']:
                raise ProtocolError('Context sent view differs from the trusted projection')
            sent = {
                'message_sha256': hashlib.sha256(args['message'].encode('utf-8')).hexdigest(),
                'message_bytes': len(args['message'].encode('utf-8')),
                'message': args['message'],
                'message_manifest': manifest,
            }
            if sent['message_bytes'] > 64 * 1024:
                raise ProtocolError('Context sent view exceeds 64 KiB')
            if len(_encode(sent)) > 1024 * 1024:
                raise ProtocolError('Context sent record exceeds its limit')
            record['sent_view'] = sent
            save(path, record)
            return {'recorded': True, 'round_id': args['round_id'],
                    'state_revision': args['state_revision']}

    def _available_profiles(self):
        current = self.source().content_hash
        return [{**r.to_dict(), 'program_ref': opaque('program', r.source_hash),
                 'source_relation': 'current' if r.source_hash == current else 'previous'}
                for r in list(self.refs.values())[-8:]]

    def _capture_for_mode(self, mode):
        if not self.pending:
            return None
        refs = [self.pending.get('clean_ref')] if mode == 'clean' else list(self.pending.get('profile_actions',{}).values())
        for key in reversed(refs):
            ref = self.refs.get(key)
            if (ref and ref.availability=='available' and ref.mode==mode
                    and ref.source_hash==self.pending['attempt']['child_source_hash']):
                return ref
        return None

    def _profile_plan(self):
        from .profile_questions import QUESTIONS
        modes = sorted({QUESTIONS[q].required_modes[0] for q in self.pending['questions'] if q in QUESTIONS}) if self.pending else []
        reused = [m for m in modes if self._capture_for_mode(m)]
        needed = [m for m in modes if m not in reused and m!='clean']
        return {'reuse_modes':reused,'needed_modes':needed,'hardware_actions_needed':len(needed),
                'budget_sufficient':self.remaining is None or len(needed)<=self.remaining,
                'clean_requires_evaluate': 'clean' in modes and 'clean' not in reused}

    def _open(self, *, hardware=False):
        if self.cancel.is_set() or self.closed.exists():
            raise ProtocolError("Method execution is closed")
        if self.intent_path.exists():
            raise ProtocolError("Interrupted public action requires reconciliation")
        if hardware and self.remaining is not None and self.remaining <= 0:
            raise ProtocolError("Public hardware budget is exhausted")

    def _action(self, operation, mode=None, question=None, purpose=None):
        self._open(hardware=True)
        action_id = str(uuid.uuid4())
        save(
            self.intent_path,
            {
                "operation": operation,
                "mode": mode,
                "purpose": purpose,
                "question": question,
                "method_action_id": action_id,
                "source_hash": self.source().content_hash,
                "hypothesis_id": (self.progress.view().active_hypothesis or self._pending_hypothesis()).hypothesis_id
                if (self.progress.view().active_hypothesis or self._pending_hypothesis())
                else None,
            },
        )
        obs = self.actions.run_action(operation, mode, cancel=self.cancel, action_id=action_id)
        self._observe_budget(obs)
        return obs

    def _observe_budget(self, obs):
        result = obs.cli_result if isinstance(obs.cli_result, dict) else {}
        if type(result.get("remaining_hardware_actions")) is int:
            self.remaining = (
                result["remaining_hardware_actions"]
                if self.remaining is None
                else min(self.remaining, result["remaining_hardware_actions"])
            )
        elif obs.operation != "fork" and not (
            result.get("remaining_hardware_actions") == "unlimited" and self.remaining is None
        ):
            # A missing public budget observation is not a license to spend again.
            save(self.closed, {"reason": "Public action omitted remaining budget"})
        if result.get("error_code") in ("session_closed", "budget_exceeded", "integrity_failed"):
            save(self.closed, {"reason": result["error_code"]})

    def call(self, name, args, call_id=None, raw_request=None):
        with self.lock:
            if call_id is not None:
                for previous in sorted((self.context.work_root / 'tool_calls').glob('*/input.json')):
                    value = json.loads(previous.read_text())
                    if value.get('call_id') != call_id:
                        continue
                    if value.get('name') != name or value.get('arguments') != args:
                        raise ProtocolError('Tool call id was reused with different content')
                    result_path = previous.with_name('result.json')
                    if not result_path.is_file():
                        raise ProtocolError('Tool call is already in progress and will not be repeated')
                    return json.loads(result_path.read_text())
            identity = uuid.uuid4().hex
            directory = self.context.work_root / "tool_calls" / identity
            before = self.source().content_hash
            h = self.progress.view().active_hypothesis or self._pending_hypothesis()
            save(
                directory / "input.json",
                {
                    "name": name,
                    "arguments": args,
                    "call_id": call_id,
                    "raw_request": raw_request,
                    "source_hash": before,
                    "hypothesis_id": h.hypothesis_id if h else None,
                },
            )
            event = self.emit(
                "tool_started",
                str((directory / "input.json").relative_to(self.context.session_root)),
                name=name,
            )
            try:
                validate_call(name, args)
                value = self._execute(name, args, call_id)
                if name == 'sluice_profile_baseline':
                    profile = self.refs.get((value.get('profile_ref') or {}).get('ref'))
                    self._register_public_evidence_refs(value.get('evidence', {}), profile)
                elif name == 'sluice_profile_detail':
                    self._register_public_evidence_refs(value, self.refs.get(args['profile_ref']))
                result = {"schema_version": 1, "is_error": False, "value": value, "context": self.view()}
            except (ProtocolError, ValueError, TypeError, KeyError) as error:
                result = {"schema_version": 1, "is_error": True, "error": str(error), "context": self.view()}
            result["event_refs"] = [event["event_id"]]
            result["source_before"] = before
            result["source_after"] = self.source().content_hash
            save(directory / "result.json", result)
            self.emit(
                "tool_finished",
                str((directory / "result.json").relative_to(self.context.session_root)),
                name=name,
            )
            self.emit(
                "decision",
                str((directory / "result.json").relative_to(self.context.session_root)),
                action=self.decision().action if self.decision() else None,
            )
            return result

    def _execute(self, name, args, call_id=None):
        if name == "sluice_progress":
            return self.view()
        if name == 'sluice_profile_catalog':
            from .profile_questions import catalog
            return catalog(args['topic'])
        if name == "sluice_search":
            if hasattr(self.knowledge, 'search_page'):
                page = self.knowledge.search_page(
                    query=args.get('query', ''),
                    observations=args.get('observations', []),
                    limit=args.get('limit', 8), topic=args.get('topic'),
                    cursor=args.get('cursor'), browse=args.get('browse', False))
                return {**page, 'items': [item.to_dict() for item in page['items']]}
            return [
                v.to_dict()
                for v in self.knowledge.search(
                    case_id=self.context.case_id,
                    source_hash=self.source().content_hash,
                    query=args["query"],
                    observations=args.get("observations", []),
                    limit=args.get("limit", 5),
                )
            ]
        if name == "sluice_read_skill":
            has_cursor = 'cursor' in args
            if has_cursor and ('offset' in args or args.get('paginate', False)):
                raise ProtocolError('Cursor reads cannot be combined with legacy offset pagination')
            if (hasattr(self.knowledge, 'read_page') and not args.get('paginate', False)
                    and args.get('offset', 0) == 0):
                return self.knowledge.read_page(args['ref'], max_bytes=args.get('max_bytes', 12000),
                                                cursor=args.get('cursor')).to_dict()
            return self.knowledge.read(
                args["ref"], max_bytes=args.get("max_bytes", 12000), offset=args.get("offset", 0),
                **({"paginate": True} if args.get("paginate") else {})
            ).to_dict()
        if name == "sluice_finish":
            state = self.progress.view()
            if state.active_hypothesis:
                raise ProtocolError("Complete evaluate and registered evidence questions before finishing")
            if state.decision is None or state.decision.action != 'stop':
                raise ProtocolError('Submit a stop Decision through sluice_step before finishing')
            save(self.closed, {"reason": args["reason"]})
            return {"finish_requested": True, "reason": args["reason"]}
        if name == 'sluice_select_program':
            return self._select_program(args, call_id=call_id)
        if name == 'sluice_step':
            return self._step(args, call_id=call_id)
        self._open()
        if name == "sluice_register":
            return self._register(args, call_id=call_id)
        if name == "sluice_evaluate":
            return self._evaluate(args)
        if name == "sluice_profile":
            return self._profile(args)
        if name == 'sluice_profile_baseline':
            return self._profile_baseline(args)
        if name == 'sluice_profile_raw':
            from .raw_profiles import read
            return read(self, **args)
        if name == 'sluice_profile_detail':
            if args['profile_ref'] not in self.refs or not hasattr(self.evidence,'details'):
                raise ProtocolError('Unknown profile reference or unsupported detail interface')
            try:
                return self.evidence.details(**args)
            except OSError as exc:
                raise ProtocolError('Evidence detail reference unavailable') from exc
        if name == "sluice_fork":
            if self.progress.view().active_hypothesis:
                raise ProtocolError("Fork requires a completed hypothesis")
            obs = self._action("fork")
            self.intent_path.unlink()
            return obs.to_dict()
        raise ProtocolError("Unknown tool")

    def _register(self, args, *, call_id=None):
        self._open()
        view = self.progress.view()
        if args['expected_revision'] != view.state_revision:
            raise ProtocolError('Stale state revision; reread progress before registration')
        current = self.source()
        if args['parent_source_hash'] == opaque('program', current.content_hash):
            args = {**args, 'parent_source_hash': current.content_hash}
        if args["parent_source_hash"] != current.content_hash:
            raise ProtocolError("Parent source hash does not match")
        if current.content_hash != self.accepted_source_hash:
            raise ProtocolError("Source changed before hypothesis registration")
        if not args["change_scope"]:
            raise ProtocolError("Change scope is required")
        if hasattr(self.evidence, 'validate_question'):
            from .evidence_guard import scan
            if args['mechanism'] is not None:
                scan(args['mechanism'])
            for question in args['verification_questions']:
                scan(question)
                self.evidence.validate_question(question, 'runtime')
        for scope in args["change_scope"]:
            p = Path(scope.split(":", 1)[0])
            if (
                p.is_absolute()
                or any(
                    c in ("..", ".", "sealed", "sealed_assets", "evaluator", "launcher", "control", "golden")
                    or c.startswith(".")
                    for c in p.parts
                )
                or not p.parts
            ):
                raise ProtocolError("Change scope must be a public relative Candidate path")
            if p.suffix != ".py":
                raise ProtocolError("Change scope must name Candidate Python source")
        for ref in args["knowledge_refs"]:
            present = self.knowledge.contains(ref) if hasattr(self.knowledge, "contains") else self._legacy_contains(ref)
            if not present:
                raise ProtocolError("Unknown knowledge reference")
        if view.decision is None or view.decision.action not in ('optimize', 'recheck'):
            raise ProtocolError('Submit an optimize or recheck Decision through sluice_step before registration')
        if view.decision.opportunity_id not in {op.opportunity_id for op in view.opportunities}:
            raise ProtocolError('Decision must reference a registered opportunity')
        opportunity = next(op for op in view.opportunities if op.opportunity_id == view.decision.opportunity_id)
        if opportunity.status not in ('open', 'needs_evidence', 'recheck'):
            raise ProtocolError('Decision opportunity is not open for exploration')
        if opportunity.mechanism_id:
            ref = opportunity.mechanism_id if ':' in opportunity.mechanism_id else 'card:' + opportunity.mechanism_id
            if not self.knowledge.contains(ref):
                raise ProtocolError('Opportunity mechanism is not in the published knowledge package')
        hypothesis_args = {key: value for key, value in args.items() if key != 'expected_revision'}
        hypothesis_args['opportunity_id'] = view.decision.opportunity_id
        hypothesis_args['mechanism_id'] = opportunity.mechanism_id
        hypothesis_args['program_ref'] = opaque('program', current.content_hash)
        hypothesis_args['kind'] = view.decision.action
        h = Hypothesis(
            "h-%04d" % (len(view.attempts) + 1),
            self.context.case_id,
            **{k: tuple(v) if isinstance(v, list) else v for k, v in hypothesis_args.items()},
        )
        if view.active_hypothesis:
            raise ProtocolError("A hypothesis is already active")
        source_ref = self._archive_source(current.content_hash)
        save(self.source_path, current.to_dict())
        self.progress.register_hypothesis(h, source_ref=source_ref, call_id=call_id)
        path = self.context.work_root / "hypotheses" / f"{h.hypothesis_id}.json"
        save(path, h.to_dict())
        self.emit("hypothesis_registered", str(path.relative_to(self.context.session_root)))
        return self.view()["progress"]

    def _step(self, args, *, call_id=None):
        self._open()
        state = self.progress.view()
        if args['expected_revision'] != state.state_revision:
            raise ProtocolError('Stale state revision; reread progress and resubmit the Decision')
        if state.active_hypothesis or self.pending or self.intent_path.exists():
            raise ProtocolError('A Decision can be submitted only after the active cycle is complete')
        completed_cycle_id = args.get('completed_cycle_id')
        completed_cycle = None
        if completed_cycle_id:
            completed_cycle = next((item for item in state.cycles
                                    if completed_cycle_id in (item.cycle_id, opaque('cycle', item.cycle_id))), None)
            if completed_cycle is None or completed_cycle.status not in ('feedback_ready', 'completed'):
                raise ProtocolError('The referenced cycle is not ready for interpretation')
            # ContextRenderer exposes a typed opaque cycle reference, while
            # sluice_progress exposes the journal cycle_id. Accept either at
            # the tool boundary and keep the canonical journal identity inside.
            completed_cycle_id = completed_cycle.cycle_id
        opportunities = args.get('opportunities', [])
        normalized_opportunities = []
        for raw_item in opportunities:
            item = dict(raw_item)
            cycle_refs = []
            for cycle_ref in item.get('cycle_refs', []):
                cycle = next((known for known in state.cycles
                              if cycle_ref in (known.cycle_id, opaque('cycle', known.cycle_id))), None)
                if cycle is None:
                    raise ProtocolError('Opportunity cites a cycle outside this Session')
                cycle_refs.append(cycle.cycle_id)
            if 'cycle_refs' in item:
                item['cycle_refs'] = cycle_refs
            normalized_opportunities.append(item)
        opportunities = normalized_opportunities
        known_evidence = {bundle.evidence_ref for bundle in state.evidence}
        known_evidence.update(ref for bundle in state.evidence for ref in bundle.provenance)
        known_evidence.update(self._known_public_evidence_refs(state))
        known_profiles = set(self.refs)
        known_programs = {program.program_ref for program in state.programs}

        def known_knowledge(ref):
            if not isinstance(ref, str):
                return False
            try:
                return (self.knowledge.contains(ref) if hasattr(self.knowledge, 'contains')
                        else self._legacy_contains(ref))
            except Exception:
                return False

        known_cycles = {cycle.cycle_id for cycle in state.cycles}
        seen = set()
        for item in opportunities:
            opportunity_id = item['opportunity_id']
            if opportunity_id in seen:
                raise ProtocolError('Opportunity IDs must be unique')
            seen.add(opportunity_id)
            refs = item.get('evidence_refs', [])
            if any(ref not in known_evidence and ref not in known_profiles for ref in refs):
                raise ProtocolError('Opportunity cites evidence outside this Session')
            if any(ref not in known_cycles for ref in item.get('cycle_refs', [])):
                raise ProtocolError('Opportunity cites a cycle outside this Session')
            assessment = item.get('assessment', {})
            if set(assessment) - {'significance', 'evidence_strength', 'applicability',
                                  'exploration', 'investigation_cost'}:
                raise ProtocolError('Opportunity assessment has unknown dimensions')
            for dimension in assessment.values():
                if any(ref not in known_evidence and ref not in known_profiles and ref not in known_programs
                       and not known_knowledge(ref)
                       for ref in dimension.get('refs', [])):
                    raise ProtocolError('Assessment cites an unknown evidence, program, profile, or knowledge reference')
            if item['status'] == 'improved':
                measured = [bundle for bundle in state.evidence if bundle.evidence_ref in refs
                            and bundle.status == 'available'
                            and any(fact.status in ('measured', 'derived') for fact in bundle.facts)]
                if not measured:
                    raise ProtocolError('An improved opportunity requires registered measured evidence')
            if item['status'] == 'recheck' and not item.get('condition_change'):
                raise ProtocolError('Recheck requires a changed condition')
        decision = dict(args['decision'])
        action = decision['action']
        candidate_opportunities = {op.opportunity_id: op for op in state.opportunities}
        for item in opportunities:
            candidate_opportunities[item['opportunity_id']] = item
        opportunity_id = decision.get('opportunity_id')
        if action in ('optimize', 'diagnose', 'recheck'):
            if not opportunity_id or opportunity_id not in candidate_opportunities:
                raise ProtocolError('Decision must reference a submitted or existing opportunity')
        elif opportunity_id is not None:
            raise ProtocolError('A stop Decision cannot reference an opportunity')
        if action == 'diagnose':
            from .profile_questions import QUESTIONS
            if decision.get('evidence_question') not in QUESTIONS:
                raise ProtocolError('Diagnostic Decision requires a registered evidence question')
        if action == 'recheck':
            proposed = candidate_opportunities[opportunity_id]
            if (proposed.get('status') if isinstance(proposed, dict) else proposed.status) != 'recheck':
                raise ProtocolError('Recheck Decision requires an opportunity marked recheck')
        if action == 'optimize':
            if self.remaining is not None and self.remaining <= 0:
                raise ProtocolError('Public hardware budget is exhausted; submit a stop Decision')
            proposed = candidate_opportunities[opportunity_id]
            status = proposed.get('status', 'open') if isinstance(proposed, dict) else proposed.status
            if status not in ('open', 'needs_evidence', 'recheck'):
                raise ProtocolError('Decision opportunity is not available for optimization')
        if action == 'stop' and not decision['reason'].strip():
            raise ProtocolError('Stop Decision requires a concrete reason')
        decision.update(expected_revision=state.state_revision, proposed_by='pi')
        value = {'completed_cycle_id': completed_cycle_id, 'opportunities': opportunities,
                 'decision': decision}
        updated = self.progress.submit_step(value, call_id=call_id)
        self.emit('step_accepted', str((self.context.work_root / 'progress.jsonl').relative_to(self.context.session_root)),
                  action=action)
        return {'accepted': True, 'revision': updated.state_revision,
                'decision': updated.decision.to_dict() if updated.decision else None,
                'opportunities': [item.to_dict() for item in updated.opportunities],
                'issues': [], 'context_revision': updated.state_revision}

    def _legacy_contains(self, ref):
        """Compatibility for third-party ports predating contains(); production ports implement it."""
        try:
            self.knowledge.read(ref, max_bytes=32768)
            return True
        except Exception:
            return False

    def _evaluate(self, args):
        h = self.progress.view().active_hypothesis
        if not h:
            raise ProtocolError("Register a hypothesis before evaluate")
        if self.pending:
            raise ProtocolError("Answer the pending verification questions first")
        current = self.source()
        if args['source_hash'] == opaque('program', current.content_hash):
            args = {**args, 'source_hash': current.content_hash}
        if args["source_hash"] != current.content_hash:
            raise ProtocolError("Candidate source hash does not match")
        parent = json.loads(self.source_path.read_text())
        files = lambda v: {x["path"]: x["sha256"] for x in v["files"]}
        a, b = files(parent), files(current.to_dict())
        changed = {p for p in a.keys() | b.keys() if a.get(p) != b.get(p)}
        if changed - {s.split(":", 1)[0] for s in h.change_scope}:
            raise ProtocolError("Candidate changes exceed the registered scope")
        obs = self._action("evaluate")
        attempt = self._complete_evaluate(h, obs, current.content_hash)
        self.intent_path.unlink()
        result = {"observation": obs.to_dict(), "attempt": attempt.to_dict()}
        measured = next((r for r in self.refs.values() if r.action_id == obs.method_action_id and r.mode == 'clean'), None)
        if measured:
            result['measurement_ref'] = measured.to_dict()
        if attempt.outcome in ('action_failed','compile_fail'):
            from .failure_diagnostics import reported_failure
            diagnostic=reported_failure(self.context,obs)
            if diagnostic is not None:
                result['failure_diagnostic']=diagnostic
        return result

    def _complete_evaluate(self, h, obs, current_hash):
        r = obs.cli_result if isinstance(obs.cli_result, dict) else {}
        correctness = r.get("correctness")
        latency = r.get("latency_us")
        valid = (
            obs.exit_code == 0
            and obs.parse_status == "valid"
            and r.get("integrity_passed") is True
            and correctness == "PASS"
            and type(latency) in (int, float)
            and math.isfinite(latency)
            and latency > 0
        )
        if obs.source_before != current_hash or obs.source_after != current_hash:
            outcome = "source_mismatch"
            correctness = None
            latency = None
        elif valid:
            best = self.progress.view().best
            outcome = (
                "baseline"
                if best is None
                else "improvement"
                if latency < best.latency_us
                else "regression"
                if latency > best.latency_us
                else "no_gain"
            )
        else:
            latency = None
            outcome = (
                "correctness_fail"
                if correctness == "FAIL"
                else "compile_fail"
                if r.get("error_code") == "compile_error"
                else "action_failed"
            )
        attempt = Attempt(
            "a-" + h.hypothesis_id,
            h.hypothesis_id,
            h.case_id,
            h.parent_source_hash,
            current_hash,
            obs.method_action_id,
            obs.bench_event_seq,
            obs.result_path,
            outcome,
            correctness,
            latency,
        )
        # Even a failing evaluate is a complete, distinct attempt. Evidence can only explain a valid source.
        questions = list(h.verification_questions) if valid and outcome != "source_mismatch" else []
        self.pending = {"attempt": attempt.to_dict(), "hypothesis_id": h.hypothesis_id,
                        "questions": questions, "evidence": []}
        if valid and outcome != 'source_mismatch' and hasattr(self.evidence, 'register_evaluate_action'):
            ref = self.evidence.register_evaluate_action(obs)
            self.refs[ref.ref] = ref
            if hasattr(self.evidence, 'member_refs'):
                for member in self.evidence.member_refs(ref):
                    self.refs[member.ref] = member
            self._save_refs()
            self.pending['clean_ref'] = ref.ref
        save(self.pending_path, self.pending)
        self._apply_attempt_fact(attempt, evidence_pending=bool(questions))
        if not questions:
            self.pending_path.unlink(missing_ok=True)
            self.pending = None
            self._safety_stop_if_needed()
        return attempt

    def _profile_ref(self, obs):
        if hasattr(self.evidence, 'register_profile_action'):
            ref = self.evidence.register_profile_action(obs)
            self.refs[ref.ref] = ref
            if hasattr(self.evidence,'member_refs'):
                for member in self.evidence.member_refs(ref):
                    self.refs[member.ref] = member
            self._save_refs()
            return ref
        result = obs.cli_result if isinstance(obs.cli_result, dict) else {}
        valid = (
            obs.exit_code == 0
            and obs.parse_status == "valid"
            and result.get("integrity_passed") is True
            and obs.source_before == obs.source_after
            and obs.source_before == (self.pending['attempt']['child_source_hash']
                                     if self.pending else self.source().content_hash)
        )
        return ProfileRef(
            opaque('profile', self.context.session_id + ':' + obs.method_action_id),
            self.context.case_id,
            obs.source_before,
            obs.method_action_id,
            obs.bench_event_seq,
            obs.profile_mode,
            obs.result_path,
            "available" if valid else "unavailable",
        )

    def _baseline_ref(self, obs):
        if obs.operation == 'evaluate':
            ref = self.evidence.register_evaluate_action(obs)
            self.refs[ref.ref] = ref
            if hasattr(self.evidence, 'member_refs'):
                for member in self.evidence.member_refs(ref):
                    self.refs[member.ref] = member
            self._save_refs()
            return ref
        return self._profile_ref(obs)


    def _profile_baseline(self, args):
        from .profile_questions import QUESTIONS
        from .evidence_guard import scan, encoded
        if self.pending or self.progress.view().active_hypothesis:
            raise ProtocolError('Baseline inspection requires a completed hypothesis')
        current = self.source().content_hash
        if current != self.accepted_source_hash:
            raise ProtocolError('Source changed before hypothesis registration')
        if not self.evidence.available or not hasattr(self.evidence, 'observe'):
            raise ProtocolError('Baseline evidence is unavailable')
        spec = QUESTIONS.get(args['question'])
        if spec is None:
            raise ProtocolError('Unsupported baseline question')
        mode = spec.required_modes[0]
        if mode != 'clean' and mode not in self.context.profiling_modes:
            raise ProtocolError('Profiling mode is unavailable')
        member_refs = {member for ref in self.refs.values() for member in ref.members}
        ref = next((r for r in reversed(list(self.refs.values()))
                    if r.ref not in member_refs and r.source_hash == current
                    and r.mode == mode and r.availability == 'available'), None)
        reused = ref is not None
        if ref is None:
            if mode == 'clean' and not hasattr(self.evidence, 'register_evaluate_action'):
                raise ProtocolError('Clean evidence is unavailable')
            obs = self._action('evaluate' if mode == 'clean' else 'profile',
                               None if mode == 'clean' else mode, purpose='baseline')
            ref = self._baseline_ref(obs)
            self.intent_path.unlink()
        public_ref = ref.to_dict()
        evidence_budget = min(3800,4096-len(encoded(public_ref))-256)
        value = {'evidence': self.evidence.observe(question=args['question'], after=ref, budget_bytes=evidence_budget),
                 'profile_ref': public_ref, 'reused': reused,
                 'hardware_actions_used': 0 if reused else 1}
        scan(value)
        if len(encoded(value)) > 4096:
            raise ProtocolError('Baseline evidence exceeded its byte budget')
        return value

    def _profile(self, args):
        if not self.pending or args["question"] not in self.pending["questions"]:
            raise ProtocolError("Profile question must belong to the evaluated active hypothesis")
        if self.source().content_hash != self.pending["attempt"]["child_source_hash"]:
            raise ProtocolError("Source changed after evaluate")
        h = self.progress.view().active_hypothesis or self._pending_hypothesis()
        if h is None:
            raise ProtocolError('Registered hypothesis record is unavailable')
        supported = True
        from .profile_questions import QUESTIONS
        default_mode = QUESTIONS[args['question']].required_modes[0] if args['question'] in QUESTIONS else 'runtime'
        mode = args.get('mode', default_mode)
        if hasattr(self.evidence, 'bind'):
            self.evidence.bind(h, self.pending['attempt'])
            supported = self.evidence.validate_question(args['question'], mode)
        before = None
        if "before_ref" in args:
            before = self.refs.get(args["before_ref"])
            if before is None:
                raise ProtocolError("Unknown profile reference")
        after = self.recovered_profiles.get(args["question"])
        if after is None:
            after = self.refs.get(self.pending.get('profile_actions', {}).get(args['question']))
        if after is not None and (after.mode != mode):
            after = None
        if after is None and mode == 'clean':
            after = self.refs.get(self.pending.get('clean_ref'))
        if after is None:
            after = self._capture_for_mode(mode)
            if after is not None:
                self.pending.setdefault('profile_actions',{})[args['question']] = after.ref
                save(self.pending_path,self.pending)
        if "after_ref" in args:
            after = self.refs.get(args["after_ref"])
            if (after is None or after.source_hash != self.pending["attempt"]["child_source_hash"]
                    or after.mode != mode):
                raise ProtocolError("Unknown or mismatched after profile reference")
        h = self.progress.view().active_hypothesis or self._pending_hypothesis()
        if before is not None and before.source_hash != h.parent_source_hash:
            raise ProtocolError("Before profile does not match the hypothesis parent source")
        if before is not None and (before.mode != mode):
            raise ProtocolError('Before profile mode does not match the request')
        missing_capture = self.evidence.available and after is None and self.remaining == 0 and mode != 'clean'
        if self.evidence.available and supported and after is None and not missing_capture and mode != 'clean':
            obs = self._action("profile", mode, question=args["question"])
            after = self._profile_ref(obs)
            self.refs[after.ref] = after
            self._save_refs()
            self.pending.setdefault('profile_actions', {})[args['question']] = after.ref
            save(self.pending_path, self.pending)
        if after is not None and after.availability != "available":
            bundle = EvidenceBundle(
                args["question"],
                status="unavailable",
                missing=(after.reason or "profile_action_failed",),
                provenance=(after.ref,),
            )
        elif not supported:
            bundle = EvidenceBundle(args['question'], missing=('mode_unsupported',))
        elif missing_capture:
            bundle = EvidenceBundle(
                args["question"], status="unavailable", missing=("public_budget_exhausted",)
            )
        else:
            bundle = self.evidence.query(
                question=args["question"], before=before, after=after, budget_bytes=4096
            )
        # Legacy injected ports are projected through the same public boundary.
        bundle = self._public_bundle(bundle, h, after)
        from .evidence_guard import encoded
        if len(encoded(bundle.to_dict())) > 4096:
            raise ProtocolError("EvidencePort exceeded its byte budget")
        if bundle.question != args["question"]:
            raise ProtocolError("Evidence question mismatch")
        self.pending["evidence"].append(bundle.to_dict())
        self.pending["questions"].remove(args["question"])
        save(self.pending_path, self.pending)
        self.recovered_profiles.pop(args['question'], None)
        state = self.progress.view()
        cycle = next((item for item in reversed(state.cycles)
                      if item.hypothesis_id == h.hypothesis_id), None)
        if cycle is None:
            raise ProtocolError('Evidence cycle was not registered')
        complete = not self.pending["questions"]
        self.progress.add_evidence(
            cycle.cycle_id, bundle, complete=complete,
            call_id='evidence:' + cycle.cycle_id + ':' + opaque('question', bundle.question),
        )
        self._archive_attempt(Attempt.from_dict(self.pending['attempt']))
        if not self.pending["questions"]:
            self.pending_path.unlink(missing_ok=True)
            self.pending = None
            self._safety_stop_if_needed()
        if self.intent_path.exists():
            self.intent_path.unlink()
        return {"evidence": bundle.to_dict(), "profile_ref": after.to_dict() if after else None}

    def _apply_attempt_fact(self, attempt, *, evidence_pending=False):
        source_ref = self._archive_source(attempt.child_source_hash)
        observation = next((event for event in self.ledger.events()
                            if event.get('method_action_id') == attempt.action_id), None)
        snapshot_id = None
        result_path = self.context.session_root / attempt.observation_ref
        if result_path.is_file():
            record = json.loads(result_path.read_text())
            action_result = record.get('cli_result') or {}
            snapshot_id = action_result.get('snapshot_id')
        self.progress.apply_attempt(attempt, (), source_ref=source_ref,
                                    snapshot_id=snapshot_id, evidence_pending=evidence_pending,
                                    call_id='attempt:' + attempt.attempt_id)
        self._archive_attempt(attempt)
        if not evidence_pending:
            self._safety_stop_if_needed()

    def _safety_stop_if_needed(self):
        if self.pending or self.progress.view().active_hypothesis:
            return
        if self.remaining is None or self.remaining > 0:
            return
        state = self.progress.view()
        if state.decision and state.decision.action == 'stop':
            return
        cycle = next((item for item in reversed(state.cycles)
                      if item.status in ('feedback_ready', 'completed')), None)
        value = {
            'completed_cycle_id': cycle.cycle_id if cycle else None,
            'opportunities': [],
            'decision': Decision(
                'stop', 'Public hardware budget is exhausted',
                expected_revision=state.state_revision, proposed_by='controller',
            ).to_dict(),
        }
        self.progress.submit_step(value, call_id='safety-stop:' + state.state_revision)

    def _apply(self):
        """Legacy recovery hook for a pending attempt not yet committed."""
        attempt = Attempt.from_dict(self.pending["attempt"])
        self._apply_attempt_fact(attempt, evidence_pending=bool(self.pending.get('questions')))
        if not self.pending.get('questions'):
            self.pending_path.unlink(missing_ok=True)
            self.pending = None

    def _save_refs(self):
        from dataclasses import asdict
        save(self.refs_path, {k: {f: v for f, v in asdict(r).items() if f != 'schema_version'} for k, r in self.refs.items()})

    def _known_public_evidence_refs(self, state=None):
        """Return evidence refs issued from registered profiles in this Session."""
        from .evidence_guard import require_ref

        state = state or self.progress.view()
        known_programs = {program.program_ref: program.source_hash for program in state.programs
                          if program.case_id == self.context.case_id}
        known = set()
        for ref, record in self.public_evidence_refs.items():
            if not isinstance(record, dict) or record.get('evidence_ref') != ref:
                continue
            try:
                require_ref(ref, 'evidence')
            except ValueError:
                continue
            profile = self.refs.get(record.get('profile_ref'))
            if (profile is None or profile.ref != record.get('profile_ref')
                    or profile.case_id != self.context.case_id
                    or profile.availability != 'available'
                    or profile.source_hash != record.get('source_hash')
                    or known_programs.get(record.get('program_ref')) != profile.source_hash
                    or record.get('case_id') != self.context.case_id):
                continue
            known.add(ref)
        return known

    def _register_public_evidence_refs(self, value, profile):
        """Persist public refs returned by baseline/detail reads for a known profile."""
        from .evidence_guard import require_ref

        if (profile is None or self.refs.get(profile.ref) != profile
                or profile.case_id != self.context.case_id or profile.availability != 'available'):
            return
        state = self.progress.view()
        known_programs = {program.program_ref: program.source_hash for program in state.programs
                          if program.case_id == self.context.case_id}
        program_ref = opaque('program', profile.source_hash)
        if known_programs.get(program_ref) != profile.source_hash:
            return

        refs = set()

        def collect(item):
            if isinstance(item, dict):
                for child in item.values():
                    collect(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)
            elif isinstance(item, str) and item.startswith('evidence:'):
                try:
                    refs.add(require_ref(item, 'evidence'))
                except ValueError:
                    pass

        collect(value)
        if not refs:
            return
        for ref in refs:
            record = {
                'evidence_ref': ref,
                'case_id': self.context.case_id,
                'profile_ref': profile.ref,
                'source_hash': profile.source_hash,
                'program_ref': program_ref,
            }
            existing = self.public_evidence_refs.get(ref)
            if existing is None or existing == record:
                self.public_evidence_refs[ref] = record
        save(self.public_evidence_refs_path, self.public_evidence_refs)

    def _public_bundle(self, bundle, hypothesis, after):
        from .evidence_guard import scan
        ref = opaque('evidence', (
            hypothesis.hypothesis_id,
            self.pending['attempt']['child_source_hash'],
            bundle.question,
        ))
        facts = tuple(replace(f, source=ref, action_id=None) for f in bundle.facts)
        bundle = replace(bundle, facts=facts, evidence_ref=ref,
            status='available' if any(f.status in ('measured', 'derived') for f in facts) else 'unavailable',
            mechanism_id=hypothesis.mechanism_id, hypothesis_ref=opaque('hypothesis', hypothesis.hypothesis_id),
            program_ref=opaque('program', self.pending['attempt']['child_source_hash']),
            provenance=tuple(p if p.startswith('profile:') else opaque('profile', p) for p in bundle.provenance),
            comparison='unavailable' if bundle.comparison == 'unproven' else bundle.comparison)
        try:
            scan(bundle.to_dict())
        except ValueError:
            bundle = replace(bundle, facts=(), status='unavailable', missing=('public_evidence_rejected',), comparison='unavailable')
        return bundle

    def _archive_attempt(self, attempt, *, clear_pending=False):
        state_attempt = next((item for item in self.progress.view().attempts
                              if item.attempt_id == attempt.attempt_id), attempt)
        if attempt.correctness == 'PASS' and attempt.outcome not in ('source_mismatch', 'action_failed'):
            self.accepted_source_hash = attempt.child_source_hash
        path = self.context.work_root / "attempts" / f"{attempt.attempt_id}.json"
        save(path, state_attempt.to_dict())
        ref = str(path.relative_to(self.context.session_root))
        if not any(
            event["event_type"] == "attempt_applied" and event.get("payload_ref") == ref
            for event in self.ledger.events()
        ):
            self.emit("attempt_applied", ref)
        if clear_pending:
            self.pending_path.unlink(missing_ok=True)
            self.pending = None
