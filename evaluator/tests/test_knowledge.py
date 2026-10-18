"""Reference bundles remain method-owned, pinned and independently copied."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

BENCHMARK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCHMARK), str(BENCHMARK / 'evaluator')]

from modes._shared import knowledge
from modes._shared.config import ConfigError
import integrity


class KnowledgeTests(unittest.TestCase):
    def make_bundle(self, root):
        mode = root / 'mode'
        snapshot = mode / 'knowledge/raw_wiki'
        (snapshot / '01_pypto_lib').mkdir(parents=True)
        documents = {'README.md': b'Original reference index\n',
                     '01_pypto_lib/intro.md': b'Original programming reference\n'}
        for name, data in documents.items():
            (snapshot / name).write_bytes(data)
        manifest = {'edition': 'original', 'captured_on': '2026-10-05',
                    'pypto_revision': 'test-snapshot', 'files': [
                        {'path': name, 'sha256': hashlib.sha256(data).hexdigest()}
                        for name, data in documents.items()]}
        (snapshot / 'manifest.json').write_text(json.dumps(manifest))
        (mode / 'knowledge/knowledge.lock.json').write_text(json.dumps(knowledge.describe(snapshot)))
        return mode, snapshot

    def test_both_packaged_modes_provide_the_same_pinned_original_corpus(self):
        basic = knowledge.validate(BENCHMARK / 'modes/basic')
        cannbot = knowledge.validate(BENCHMARK / 'modes/cannbot')
        self.assertEqual(basic, cannbot)
        self.assertEqual(basic, knowledge.describe(BENCHMARK.parents[1] / 'pto_wiki/raw_wiki'))
        for mode in ('basic', 'cannbot'):
            root = BENCHMARK / 'modes' / mode / 'knowledge/raw_wiki'
            for path in [root, *root.rglob('*')]:
                self.assertFalse(path.is_symlink())
                self.assertEqual(path.stat().st_mode & 0o222, 0)

    def test_session_copies_are_independent_readonly_and_recoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mode, source = self.make_bundle(root)
            first, second = root / 'session-one', root / 'session-two'
            first_record = knowledge.freeze(mode, first)
            second_record = knowledge.freeze(mode, second)
            self.assertEqual(first_record, second_record)
            target = first / first_record['snapshot']
            other = second / second_record['snapshot']
            self.assertNotEqual((source/'README.md').stat().st_ino, (target/'README.md').stat().st_ino)
            self.assertNotEqual((other/'README.md').stat().st_ino, (target/'README.md').stat().st_ino)
            for path in [target, *target.rglob('*')]:
                self.assertEqual(path.stat().st_mode & 0o222, 0)
            protection = root / 'protection'
            integrity.capture_protection(protection, {
                'knowledge': integrity.ProtectionRoot(target, 'agent_archive', readonly=True)})
            original = (target/'README.md').read_bytes()
            (target/'README.md').chmod(0o644)
            (target/'README.md').write_text('tampered')
            report = integrity.check_and_restore(protection)
            self.assertFalse(report.passed)
            self.assertEqual((target/'README.md').read_bytes(), original)
            self.assertEqual((target/'README.md').stat().st_mode & 0o222, 0)
            self.assertEqual((source/'README.md').read_bytes(), original)
            self.assertEqual((other/'README.md').read_bytes(), original)

    def test_changed_missing_extra_and_linked_documents_are_rejected(self):
        for change in ('changed', 'missing', 'extra', 'link', 'manifest'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                mode, source = self.make_bundle(root)
                if change == 'changed':
                    (source/'README.md').write_text('different version')
                elif change == 'missing':
                    (source/'README.md').unlink()
                elif change == 'extra':
                    (source/'unexpected.md').write_text('unversioned content')
                elif change == 'link':
                    outside = root / 'outside.md'
                    outside.write_bytes((source/'README.md').read_bytes())
                    (source/'README.md').unlink()
                    (source/'README.md').symlink_to(outside)
                else:
                    original=json.loads((source/'manifest.json').read_text())
                    original['captured_on']='different-version'
                    (source/'manifest.json').write_text(json.dumps(original))
                with self.assertRaises(ConfigError):
                    knowledge.freeze(mode, root/'session')
                self.assertFalse((root/'session'/knowledge.SNAPSHOT).exists())


if __name__ == '__main__':
    unittest.main()
