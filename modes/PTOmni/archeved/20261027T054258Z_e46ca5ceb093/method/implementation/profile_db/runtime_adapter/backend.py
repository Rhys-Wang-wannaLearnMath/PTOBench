# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inspect the selected Simpler backend without importing device runtime code."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import os
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse


class BackendUnavailable(ValueError):
    """The selected backend cannot supply a required profiling capability."""


HOST_CONTRACTS = {
    'strace_timing': ('parse_spans','count_record_heads','load_host_phase_records',
                      'host_record_spans','parse_drop_summaries', 'invocation_spans',
                      'group_invocations', '_round_metrics'),
    'hbg_bind_phases': ('parse_binds','cold_binds','spread'),
    'phase_time_split': ('parse','cold_keys','summarise'),
}

PARSER_SECTIONS = frozenset(('scheduler_tasks', 'scheduler_records', 'aicpu_lifecycle_records'))


def selection():
    """Keep old collectors usable with one coherent owned analysis stack.

    Parser output shapes feed scheduler, critical-path and export modules.
    Upgrading only the parser would mix incompatible implementations. An old
    converter therefore selects the complete owned bundle for this process.
    No collector code or installation is changed.
    """
    requested = os.environ.get('PFDB_ANALYSIS_BACKEND', 'bundled')
    if requested not in ('bundled', 'installed'):
        raise BackendUnavailable('PFDB_ANALYSIS_BACKEND must be bundled or installed')
    reason = None
    if requested == 'installed':
        try:
            converter = importlib.import_module('simpler_setup.tools.swimlane_converter')
            reader = getattr(converter, 'read_perf_data', None)
            if not callable(reader) or not PARSER_SECTIONS <= parser_keys(reader):
                reason = 'installed converter lacks the complete scheduler/lifecycle parser contract'
            if reason is None:
                contracts = HOST_CONTRACTS
                for name, functions in contracts.items():
                    module = importlib.import_module('simpler_setup.tools.' + name)
                    if any(not callable(getattr(module, f, None)) for f in functions):
                        reason = 'installed analysis lacks the complete ' + name + ' contract'
                        break
        except ImportError:
            reason = 'installed analysis converter unavailable'
    return dict(requested=requested,
                effective='sluice_offline_bridge' if reason else requested, reason=reason)


def require(name, *functions):
    try:
        selected = selection()
        owned = selected['effective'] != 'installed'
        prefix = "profile_db._offline_tools." if owned else "simpler_setup.tools."
        mod = importlib.import_module(prefix + name)
    except ImportError as exc:
        raise BackendUnavailable(f"Simpler backend requires {name}: {exc}") from exc
    missing = [f for f in functions if not callable(getattr(mod, f, None))]
    if missing:
        raise BackendUnavailable(f"Simpler backend requires {name}: missing {', '.join(missing)}")
    return mod


def parser_keys(function):
    """Conservative preflight; output preservation is checked separately."""
    decoder = getattr(function, '__globals__', {}).get('_decode_perf_data', function)
    try:
        return _parser_keys(decoder)
    except TypeError:
        return frozenset()


@lru_cache(maxsize=64)
def _parser_keys(decoder):
    # Cache by the actual decoder object, never by install path or backend label.
    # Replacing a decoder therefore rechecks the format contract.
    try:
        tree = ast.parse(inspect.getsource(decoder))
        return frozenset(node.value for node in ast.walk(tree)
                         if isinstance(node, ast.Constant) and isinstance(node.value, str))
    except (OSError, TypeError, SyntaxError):
        return frozenset()


def _converter_provenance():
    """Never assign an enclosing project's HEAD to an installed wheel.

    A local-source install may identify its original checkout only when the
    installed converter bytes still match that checkout. Function digests in
    each snapshot identify the actual analysis implementation independently.
    """
    try:
        converter = require("swimlane_converter", "read_perf_data")
        path = Path(converter.__file__).resolve()
    except (BackendUnavailable, TypeError):
        return dict(revision=None, converter_sha256=None, source="unavailable")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    owned = Path(__file__).resolve().parents[1] / "_offline_tools"
    if path.parent == owned:
        notice = json.loads((owned / "NOTICE.json").read_text())
        actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in owned.glob("*.py")
                  if p.name != "__init__.py"}
        clean = actual == {k: v["sha256"] for k, v in notice["files"].items()}
        return dict(revision=notice["revision"], converter_sha256=sha,
                    source="sluice_offline_bundle", dirty=not clean,
                    bundle_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest())
    candidates = [(path.parents[2], "checkout")]
    try:
        dist = importlib.metadata.distribution("simpler")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        url = urlparse(direct.get("url", ""))
        if url.scheme == "file":
            candidates.append((Path(unquote(url.path)).resolve(), "local_install"))
    except (importlib.metadata.PackageNotFoundError, ValueError):
        pass
    for root, source in candidates:
        original = root / "simpler_setup/tools/swimlane_converter.py"
        if not (root / ".git").exists() or not original.is_file():
            continue
        if hashlib.sha256(original.read_bytes()).hexdigest() != sha:
            continue
        try:
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--", "simpler_setup/tools"],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        return dict(revision=revision, converter_sha256=sha, source=source, dirty=bool(dirty))
    return dict(revision=None, converter_sha256=sha, source="installed_copy")


def provenance():
    result = _converter_provenance()
    hashes = {}
    for name, contract in HOST_CONTRACTS.items():
        try:
            path = Path(require(name, *contract).__file__)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (BackendUnavailable, AttributeError, TypeError, OSError):
            hashes[name] = None
    # Function bodies alone do not identify regexes/constants/helper globals.
    result['host_module_sha256'] = hashes
    return result


def diagnostics():
    """Report supported features explicitly; absence never means zero work."""
    features = {}
    checks = {
        "clock_join": ("swimlane_converter", ("read_perf_data",)),
        "host": ("strace_timing", ("parse_spans", "count_record_heads", "load_host_phase_records",
                                   "host_record_spans", "parse_drop_summaries")),
        "merged": ("swimlane_converter", ("_generate_l3_trace", "_load_dispatch_identity")),
    }
    for name, (mod, functions) in checks.items():
        try:
            require(mod, *functions)
            if name == "host":
                require("hbg_bind_phases", "parse_binds", "cold_binds", "spread")
                require("phase_time_split", "parse", "cold_keys", "summarise")
            features[name] = dict(available=True)
            if name == 'host':
                features[name]['module_sources'] = {
                    module: (('sluice_offline_bridge' if os.environ.get('PFDB_ANALYSIS_BACKEND') == 'installed' else 'sluice_offline_bundle')
                        if require(module).__name__.startswith('profile_db.') else 'installed')
                    for module in HOST_CONTRACTS}
                features[name]['function_provenance'] = 'exact implementation hashes in each analysis snapshot'
        except BackendUnavailable as exc:
            features[name] = dict(available=False, reason=str(exc))
    try:
        converter = require("swimlane_converter", "read_perf_data")
        keys = parser_keys(converter.read_perf_data)
        for key in ("scheduler_tasks", "scheduler_records", "aicpu_lifecycle_records"):
            features[key] = dict(available=key in keys)
    except BackendUnavailable:
        for key in ("scheduler_tasks", "scheduler_records", "aicpu_lifecycle_records"):
            features[key] = dict(available=False)
    return dict(selection=selection(), backend=provenance(), capabilities=features)
