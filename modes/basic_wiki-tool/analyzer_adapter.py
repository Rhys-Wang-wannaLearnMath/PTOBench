"""Session-local selection and routing; all analysis stays in the frozen copy."""
import copy
import json
from pathlib import Path
import threading

import session
from pto_analyzer import ProfileRef, ProfileRefError, open_toolset
from pto_analyzer.profile import ProfileArchive

MANIFEST = Path(__file__).with_name('analyzer_manifest.json')


class AnalyzerAdapter:
    def __init__(self, session_root: Path):
        self.root = Path(session_root).resolve(strict=True)
        self._manifest = json.loads(MANIFEST.read_text())
        self._cached_ref = None
        self._groups = {}
        self._authority = None
        self._lock = threading.Lock()

    def specifications(self) -> list[dict]:
        return copy.deepcopy(self._manifest['specifications'])

    def call(self, name: str, arguments: dict) -> dict:
        if name not in self._manifest['routing']:
            raise KeyError(f'unknown PTOAnalyzer tool: {name}')
        if not isinstance(arguments, dict):
            raise ValueError('tool arguments must be an object')
        args = arguments.copy()
        explicit = 'profile' in args
        identity = args.pop('profile', None)
        if 'cursor' in args and not explicit:
            raise ValueError('cursor requires an explicit profile event ID')
        with self._lock:
            with session._session_lock(self.root):
                metadata = json.loads((self.root / 'session.json').read_text())
                events = [json.loads(line) for line in (self.root / 'events.jsonl').read_text().splitlines() if line.strip()]
                if explicit:
                    prefix = metadata['session_id'] + ':'
                    if (not isinstance(identity, str) or not identity.startswith(prefix)
                            or not identity[len(prefix):].isascii()
                            or not identity[len(prefix):].isdigit()):
                        raise ValueError('profile must be a current Session event ID (session_id:event_seq)')
                    candidates = [e for e in events if e.get('seq') == int(identity[len(prefix):])]
                else:
                    candidates = sorted(events, key=lambda e: e.get('seq', 0), reverse=True)
                selected = None
                for event in candidates:
                    if (event.get('action') != 'profile' or event.get('mode') != 'combined'
                            or event.get('state') != 'completed' or event.get('integrity_passed') is not True
                            or event.get('case_id', metadata['case_id']) != metadata['case_id']):
                        continue
                    try:
                        ref = ProfileRef.from_event(self.root, event['seq'])
                        authorized = ProfileArchive(ref)
                        authority = (authorized.artifact_root, authorized._entries, authorized._protected_entries)
                        role = self._manifest['routing'][name]
                        if ref != self._cached_ref or authority != self._authority:
                            self._groups.clear()
                            self._cached_ref = None
                        if role not in self._groups:
                            self._groups[role] = open_toolset(role, ref)
                        self._cached_ref = ref
                        self._authority = authority
                        selected = self._groups[role]
                        break
                    except ProfileRefError:
                        if explicit:
                            raise
                if selected is None:
                    raise ValueError('No authorized completed combined Profile; run `ptobench profile .` first, or select an existing Session profile ID.')
            try:
                return selected.call(name, args)
            except Exception as error:
                error.profile = selected.profile
                raise
