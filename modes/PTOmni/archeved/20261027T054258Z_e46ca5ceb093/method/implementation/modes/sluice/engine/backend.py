"""Reuse shared profiles, provider configuration, readiness and native recording."""

from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
# The shared library uses PTOBench's public support-module import convention.
for path in (ROOT, ROOT / "evaluator"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modes._shared import provider, recording
from modes._shared.adapters import get_adapter
from modes._shared.config import load_profile, render
import common
from ..foundation.protocol import ProtocolError

MODE = Path(__file__).resolve().parents[1]
DEFAULT_ALLOW_RAW_PROFILE_READ = True
DEFAULT_STAGE01_SMOKE_ACCEPTANCE = False
TRANSPORT_STATUS = {
    "pi": "native_extension_loader_and_local_tools_verified",
    "codex": "native_config_and_synthetic_stdio_verified_model_session_unverified",
}


def preflight(experiment):
    options = experiment.get('method_options', {})
    if not isinstance(options, dict) or set(options) - {
            'evidence_port', 'allow_raw_profile_read', 'stage01_smoke_acceptance'}:
        raise ProtocolError('Unsupported SLUICE method option')
    if type(options.get('allow_raw_profile_read', DEFAULT_ALLOW_RAW_PROFILE_READ)) is not bool:
        raise ProtocolError('allow_raw_profile_read must be a boolean')
    if type(options.get('stage01_smoke_acceptance', DEFAULT_STAGE01_SMOKE_ACCEPTANCE)) is not bool:
        raise ProtocolError('stage01_smoke_acceptance must be a boolean')
    if options.get('evidence_port', 'unavailable') not in ('unavailable', 'profile_db'):
        raise ProtocolError('Unsupported evidence port')
    opt = experiment["optimizer"]
    backend = opt["backend"]
    if backend == "fake":
        return {
            "schema_version": 1,
            "profile": {"backend": "fake", "name": opt["profile"], "test_only": True},
            "backend": {"backend": "fake", "version": "synthetic-v1"},
            "optimizer_metadata": {
                "command": [sys.executable, "-m", "modes.sluice.engine.fake_agent"],
                "tool_version": "synthetic-v1",
            },
        }
    if backend not in ("pi", "codex"):
        raise ProtocolError("SLUICE supports Pi and Codex native backends")
    # The selected shared Pi profile owns model and reasoning settings. Mode
    # .env remains responsible for the gateway and credentials.
    profile = recording.prepare_profile(provider.configure(
        load_profile(backend, opt["profile"]), MODE, prefer_profile=True
    ))
    info = get_adapter(backend).preflight(profile)
    return {
        "schema_version": 1,
        "profile": profile,
        "backend": info.to_dict(),
        "optimizer_metadata": {"command": profile["argv"], "tool_version": info.version},
        "transport_validation": TRANSPORT_STATUS,
    }


def command(config, prompt):
    context = config["context"]
    root = Path(context["session_root"])
    workspace = Path(context["workspace"])
    directory = root / "method/agent/main"
    helper = MODE / 'engine' / ('tools_server.py' if config['profile']['backend'] == 'fake' else 'tool_relay.py')
    environment = {
        "SLUICE_CONFIG": str(root / "method/config.json"),
        "SLUICE_PYTHON": sys.executable,
        "SLUICE_TOOLS": str(helper),
        "PTOBENCH_SESSION_ROOT": str(root),
        "PTOBENCH_SESSION_ID": context["session_id"],
        "PTOBENCH_CASE_ID": context["case"]["case_id"],
        "PTOBENCH_WORKSPACE": str(workspace),
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                (
                    str(ROOT),
                    str(common.BENCHMARK_ROOT),
                    str(common.BENCHMARK_ROOT / "evaluator"),
                    os.environ.get("PYTHONPATH"),
                ),
            )
        ),
    }
    if config["profile"]["backend"] == "fake":
        return [sys.executable, str(MODE / "engine/fake_agent.py")], environment
    profile = config["profile"]
    if profile["backend"] == "codex":
        environment["CODEX_HOME"] = str(directory / "agent/codex")
    argv, additions = render(profile, workspace, directory, prompt)
    if additions.get("PYTHONPATH"):
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(ROOT),
                str(common.BENCHMARK_ROOT),
                str(common.BENCHMARK_ROOT / "evaluator"),
                additions["PYTHONPATH"],
            )
        )
    if profile["backend"] == "pi":
        options = ["--extension", str(MODE / "bridge/pi_sluice.ts")]
    else:
        options = []
        for key, value in {
            "mcp_servers.sluice.command": sys.executable,
            "mcp_servers.sluice.args": [str(helper), "--config", str(root / "method/config.json")],
        }.items():
            options += ["-c", key + "=" + json.dumps(value)]
    index = profile["argv"].index("{prompt}")
    argv[index:index] = options
    return argv, {**additions, **environment}
