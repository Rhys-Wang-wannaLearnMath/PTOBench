"""Interpret the diagnostic copy after a successful public profile action.

Legacy public results carry execution-workspace locators. Translate that metadata
without opening the execution workspace. Capture parsing remains in ProfileDB.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

from .context import _has_symlink_component

MODES = {'runtime': 'runtime', 'pmu': 'hardware', 'compiler': 'compile'}


def _read(path):
    if _has_symlink_component(path):
        raise ValueError('public_path_invalid')
    return json.loads(path.read_text())


def software_identity(context):
    """Read only launcher-public provenance; absent versions stay absent."""
    try:
        provenance = _read(context.session_root / 'environment.json')
        packages = provenance.get('packages', {})
        if all(packages.get(name, {}).get('version') for name in ('pypto', 'simpler', 'torch')):
            software = {key: provenance.get(key) for key in ('python', 'packages', 'toolchain_version_files', 'driver_version')}
            return hashlib.sha256(json.dumps(software, sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def public_result(context, observation):
    """Read a normalized metadata view without writing public diagnostic files.

    The public output tree is read-only in an isolated method. Translation is
    pure and repeated on recovery; only session-local handles are persisted.
    """
    clean = observation.operation == 'evaluate' and observation.profile_mode is None
    if ((not clean and (observation.operation != 'profile' or observation.profile_mode not in MODES))
            or observation.exit_code != 0 or observation.parse_status != 'valid'):
        raise ValueError('profile_action_failed')
    result = observation.cli_result or {}
    if result.get('status') != 'PASS' or result.get('integrity_passed') is not True:
        raise ValueError('profile_action_failed')
    workspace = context.validate_workspace(Path(observation.workspace))
    mode = 'clean' if clean else MODES[observation.profile_mode]
    root = workspace / '.benchmark_outputs' / mode
    public = _read(root / 'latest_session_result.json')
    if any(public.get(k) != result.get(k) for k in ('status', 'event_seq', 'integrity_passed')):
        raise ValueError('action_result_mismatch')
    raw = _read(root / 'latest_result.json')
    if raw.get('case_id') != context.case_id or raw.get('profiling_mode') != mode:
        raise ValueError('identity_mismatch')
    try:
        artifact = Path(raw['artifact_dir'])
        if artifact != root:
            # Match only the existing public execution-layout spelling; the
            # path is metadata, and is never opened or traversed.
            relative = artifact.relative_to(context.session_root).as_posix()
            if not re.fullmatch(r'staging/action_[0-9]{6}/workspace/\.benchmark_outputs/' + mode, relative):
                return raw
            def translate(value):
                if value is None:
                    return None
                build = Path(value).relative_to(artifact)
                translated = root / build
                if ('..' in build.parts or _has_symlink_component(translated)
                        or (not clean and not translated.is_dir())):
                    raise ValueError('public_path_invalid')
                return str(translated)
            selected = translate(raw.get('build_dir'))
            manifest = [{**item, 'build_dir':translate(item['build_dir'])}
                        for item in raw.get('build_manifest', [])]
            raw.update(artifact_dir=str(root), build_dir=selected)
            if 'build_manifest' in raw:
                raw['build_manifest'] = manifest
    except (ValueError, KeyError, TypeError):
        return raw
    if raw.get('software_identity') is None:
        try:
            # Preparation preserves this public identity before the supervisor
            # hides host invocation records. The method record is protected.
            identity = _read(context.method_root / 'profile_context.json').get('software_identity')
        except (OSError, ValueError, KeyError, TypeError):
            identity = software_identity(context)
        if identity is not None:
            raw['software_identity'] = identity
    return raw
