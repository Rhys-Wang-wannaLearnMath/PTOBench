"""PTOBench lifecycle operations, keeping all method changes under SLUICE."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
from . import backend
from .agent import NativeAgent
from .controller import Controller
from .contracts import digest
from .evidence_port import UnavailableEvidencePort
from .knowledge_port import PublishedKnowledgePort, bound_knowledge, sha
from .implementation import freeze
from .progress_port import JournalProgressPort
from .prompts import main_prompt
from ..foundation.context import RunContext
from ..foundation.events import EventLedger
from ..foundation.protocol import ProtocolError, response
from ..foundation.resources import (
    _resource_manifest,
    _readonly,
    _write,
    collect as foundation_collect,
)
from ..foundation.source import fingerprint

PORTS = {
    "knowledge": PublishedKnowledgePort.version,
    "evidence": UnavailableEvidencePort.version,
    "progress": JournalProgressPort.version,
    "schema_version": 1,
    "provisional": True,
}


def readonly_tree(root):
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_symlink():
            raise ProtocolError("Archive cannot contain symlinks")
        _readonly(p)
    _readonly(root)


def preflight(request):
    root = knowledge_source()
    knowledge = bound_knowledge(root)
    result = backend.preflight(request["experiment"])
    options = request['experiment'].get('method_options', {})
    stage01_smoke_acceptance = options.get(
        'stage01_smoke_acceptance', backend.DEFAULT_STAGE01_SMOKE_ACCEPTANCE
    )
    allow_raw_profile_read = options.get(
        'allow_raw_profile_read', backend.DEFAULT_ALLOW_RAW_PROFILE_READ
    )
    if stage01_smoke_acceptance and not allow_raw_profile_read:
        raise ProtocolError('Stage 01 smoke acceptance requires raw profile reads to be enabled')
    # Native optimization sessions ship and use the local profiling core by
    # default. The synthetic transport backend deliberately has no captures.
    default_port = ('unavailable' if result['profile']['backend'] == 'fake' else 'profile_db')
    selection = request['experiment'].get('method_options', {}).get('evidence_port',
        os.environ.get('SLUICE_EVIDENCE_PORT', default_port))
    if selection not in ('unavailable', 'profile_db'):
        raise ProtocolError('Unsupported evidence port')
    if selection == 'profile_db':
        from .profile_core import profile_db_class
        profile_db_class()
    result['evidence_port'] = selection
    result['allow_raw_profile_read'] = allow_raw_profile_read
    result['stage01_smoke_acceptance'] = stage01_smoke_acceptance
    result["knowledge"] = {
        "knowledge_digest": knowledge.binding["knowledge_digest"],
        "environment": knowledge.binding["environment"],
    }
    return result


def knowledge_source():
    """Operator override selects a release; it never bypasses binding validation."""
    return Path(os.environ.get("SLUICE_KNOWLEDGE_ROOT", str(backend.MODE / "resources/knowledge")))


def prepare(request):
    context = RunContext.from_request(request)
    ready = request["preflight"]
    allow_raw_profile_read = ready.get('allow_raw_profile_read', backend.DEFAULT_ALLOW_RAW_PROFILE_READ)
    if (type(allow_raw_profile_read) is not bool or
            allow_raw_profile_read != request['experiment'].get('method_options', {}).get(
                'allow_raw_profile_read', backend.DEFAULT_ALLOW_RAW_PROFILE_READ
            )):
        raise ProtocolError('Raw profile read policy changed after preflight')
    stage01_smoke_acceptance = ready.get(
        'stage01_smoke_acceptance', backend.DEFAULT_STAGE01_SMOKE_ACCEPTANCE
    )
    if (type(stage01_smoke_acceptance) is not bool or
            stage01_smoke_acceptance != request['experiment'].get('method_options', {}).get(
                'stage01_smoke_acceptance', backend.DEFAULT_STAGE01_SMOKE_ACCEPTANCE
            )):
        raise ProtocolError('Stage 01 smoke acceptance policy changed after preflight')
    if stage01_smoke_acceptance and not allow_raw_profile_read:
        raise ProtocolError('Stage 01 smoke acceptance requires raw profile reads to be enabled')
    profile = ready["profile"]
    opt = request["experiment"]["optimizer"]
    if (profile["backend"], profile["name"]) != (opt["backend"], opt["profile"]):
        raise ProtocolError("Prepared profile does not match experiment")
    if profile["backend"] != "fake":
        profile = backend.provider.prepare(profile, context.session_root)
    source = knowledge_source()
    approved = bound_knowledge(source)
    if ready.get("knowledge") != {key: approved.binding[key] for key in ("knowledge_digest", "environment")}:
        raise ProtocolError("Knowledge release changed after preflight")
    resources = context.method_root / "resources/knowledge"
    if resources.parent.exists():
        raise ProtocolError("SLUICE resources are already prepared")
    shutil.copytree(
        source,
        resources,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    release_path = resources.parent / "knowledge_release.json"
    _write(release_path, approved.binding)
    knowledge = bound_knowledge(resources)
    readonly_tree(context.method_root / "resources")
    evidence_port = ready.get('evidence_port', 'unavailable')
    if evidence_port not in ('unavailable', 'profile_db'):
        raise ProtocolError('Unsupported evidence port')
    implementation = freeze(backend.MODE, context.method_root / "implementation", profile_tool=evidence_port == 'profile_db')
    readonly_tree(context.method_root / "implementation")
    controller = Controller(context, knowledge=knowledge)
    initial = controller.view()
    if evidence_port == 'profile_db':
        from .profile_questions import public_questions
        initial['profile_questions'] = public_questions()
    initial["knowledge_candidates"] = [
        h.to_dict()
        for h in knowledge.search(
            case_id=context.case_id,
            source_hash=initial["source_hash"],
            query="task granularity frontend",
            observations=[],
            limit=3,
        )
    ]
    from .public_output import project as public_project
    prompt = main_prompt(
        context,
        public_project(initial),
        stage01_smoke_acceptance=stage01_smoke_acceptance,
    )
    from .evidence_port import EvidenceFacade
    ports = {**PORTS, 'evidence': EvidenceFacade.version if evidence_port == 'profile_db' else UnavailableEvidencePort.version}
    source = fingerprint(context.workspace).to_dict()
    config = {
        "schema_version": 1,
        "core": "sluice-core-v1",
        "evidence_port": evidence_port,
        "allow_raw_profile_read": allow_raw_profile_read,
        "stage01_smoke_acceptance": stage01_smoke_acceptance,
        "context": context.public_dict(),
        "experiment": request["experiment"],
        "profile": profile,
        "backend": ready["backend"],
        "profile_sha256": profile.get("sha256"),
        "profile_content_sha256": digest(profile),
        "effective_model": ready["backend"].get(
            "model", "synthetic" if profile["backend"] == "fake" else "unknown"
        ),
        "effective_reasoning": profile.get("reasoning")
        or (
            backend.recording.codex_override(
                profile.get("argv", []), "model_reasoning_effort"
            )
            if profile["backend"] == "codex"
            else backend.recording.option(profile.get("argv", []), "--thinking")
        ),
        "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
        "knowledge_binding_sha256": sha(release_path),
        "knowledge_digest": knowledge.binding["knowledge_digest"],
        "knowledge_environment": knowledge.binding["environment"],
        "implementation_sha256": implementation["content_sha256"],
        "ports": ports,
        "source": source,
        "transport_validation": backend.TRANSPORT_STATUS,
        "test_only": profile["backend"] == "fake",
    }
    manifest = _resource_manifest(context.method_root / "resources")
    records = {
        "method/config.json": config,
        "method/prompt.txt": prompt,
        "method/ports.json": ports,
        "method/knowledge.json": {
            "knowledge_digest": config["knowledge_digest"],
            "active": sorted(knowledge.cards),
        },
        "method/resource_manifest.json": manifest,
        "method/implementation.json": implementation,
        "method/source.json": source,
        "method/README.md": "SLUICE single-session optimization. Use the registered tools and frozen knowledge package.\n",
    }
    if evidence_port == 'profile_db':
        from ..foundation.profile_publication import software_identity
        records['method/profile_context.json'] = {'software_identity': software_identity(context)}
    for name, value in records.items():
        p = context.session_root / name
        _write(p, value)
        _readonly(p)
    EventLedger(context).append(
        "run_prepared",
        source_hash=source["content_sha256"],
        payload_ref="method/config.json",
    )
    return response(
        records=records,
        archives={
            "resources": "method/resources",
            "implementation": "method/implementation",
        },
    )


def load_config(context, request=None):
    config = json.loads((context.method_root / "config.json").read_text())
    if config["context"] != context.public_dict() or (
        request and config["experiment"] != request["experiment"]
    ):
        raise ProtocolError("Prepared context does not match request")
    if digest(config["profile"]) != config["profile_content_sha256"]:
        raise ProtocolError("Prepared profile content hash mismatch")
    if sha(context.method_root / "prompt.txt") != config["prompt_sha256"]:
        raise ProtocolError("Prepared prompt hash mismatch")
    if (
        sha(context.method_root / "resources/knowledge_release.json")
        != config["knowledge_binding_sha256"]
    ):
        raise ProtocolError("Prepared knowledge binding mismatch")
    validated_environment_digest = None
    if request is not None:
        preflight_knowledge = request.get("preflight", {}).get("knowledge")
        expected_knowledge = {
            "knowledge_digest": config.get("knowledge_digest"),
            "environment": config.get("knowledge_environment"),
        }
        if preflight_knowledge != expected_knowledge:
            raise ProtocolError("Prepared knowledge identity differs from Launcher preflight")
        validated_environment_digest = expected_knowledge["environment"]
    else:
        validated_environment_digest = config.get("knowledge_environment")
    knowledge = bound_knowledge(
        context.method_root / "resources/knowledge",
        validated_environment_digest=validated_environment_digest,
    )
    if knowledge.binding["knowledge_digest"] != config["knowledge_digest"]:
        raise ProtocolError("Prepared knowledge package mismatch")
    return config


def run(request):
    context = RunContext.from_request(request)
    config = load_config(context, request)
    ledger = EventLedger(context)
    ledger.append(
        "run_started",
        source_hash=fingerprint(context.workspace).content_hash,
        payload_ref="method/config.json",
    )
    result = NativeAgent().run(context, config)
    ledger.append(
        "run_finished",
        source_hash=fingerprint(context.workspace).content_hash,
        payload_ref="method/agent/main/result.json",
    )
    return response(
        status="failed" if result["is_error"] else "completed", agent=result
    )


def collect(request):
    context = RunContext.from_request(request)
    work = context.work_root
    missing = []
    for path in (work / "tool_calls").glob("*/input.json"):
        if not path.with_name("result.json").exists():
            missing.append(str(path.parent.relative_to(context.session_root)))
    if (work / "action_intent.json").exists():
        missing.append("unreconciled_action")
    result = foundation_collect(context, request)
    summary = result["records"]["method/summary.json"]
    agent = context.method_root / "agent/main"
    native = {"status": "not_started"}
    if agent.exists():
        p = agent / "recording.json"
        if p.is_file():
            native = json.loads(p.read_text())
        else:
            native = {"status": "missing"}
        if native["status"] == "missing":
            missing.append("native_recording")
        if not (agent / "result.json").exists():
            missing.append("agent_exit")
    elif request.get("started"):
        missing.append("agent_not_recorded")
    if (work / "pending.json").exists():
        missing.append("pending_attempt")
    status = (
        "missing"
        if missing or result["recording"]["status"] == "missing"
        else "recorded"
    )
    summary = {
        **summary,
        "core": "sluice-core-v1",
        "recording": {"status": status},
        "missing": sorted(set(summary["missing"] + missing)),
        "progress": JournalProgressPort(work / "progress.jsonl").view().to_dict(),
        "backend_recording": native,
        "transport_validation": backend.TRANSPORT_STATUS,
    }
    p = context.method_root / "summary.json"
    p.chmod(0o644)
    _write(p, summary)
    _readonly(p)
    result["records"]["method/summary.json"] = summary
    if agent.exists():
        result["archives"]["sluice_agent"] = "method/agent"
    for name in result["archives"].values():
        readonly_tree(context.session_root / name)
    result["recording"] = {"status": status, "missing": summary["missing"]}
    return result
