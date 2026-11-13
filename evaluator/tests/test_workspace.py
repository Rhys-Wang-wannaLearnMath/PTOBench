from __future__ import annotations

import sys
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
import grp


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATOR_DIR))
import workspace
import common
import integrity


class WorkspaceDocumentationTest(unittest.TestCase):
    def test_public_guide_covers_visible_files_commands_and_feedback_only(self):
        guide = workspace._readme({'case_id': 'test.case', 'module': 'candidate.py', 'entry': 'entry'},
                                  workspace.profile_modes.public_enabled_modes('combined'))
        for filename in ('case.json', 'config.py', 'comparator.json', 'reference.json',
                         'environment.json', 'workspace.json',
                         'starting_point/', 'latest_result.json', 'latest_session_result.json',
                         'tuning.json'):
            self.assertIn(filename, guide)
        self.assertNotIn('mode.json', guide)
        self.assertNotIn('Do not inspect hidden test/reference assets', guide)
        self.assertIn('candidate/candidate.py', guide)
        self.assertIn('candidate/tuning.json', guide)
        self.assertIn('Candidate Python files and local helpers', guide)
        self.assertIn('read-only result and diagnostic copies', guide)
        for command in ('evaluate', 'profile', 'fork'):
            self.assertIn('ptobench ' + command, guide)
        for path in ('clean/', 'combined/'):
            self.assertIn('.benchmark_outputs/' + path, guide)
        for field in ('status', 'error_code', 'message', 'latency_us', 'remaining_hardware_actions'):
            self.assertIn(field, guide)
        self.assertIn(
            'Profile does not check output correctness. Use `ptobench evaluate` to check correctness and clean latency.',
            guide,
        )
        documents = {
            'workspace README': guide,
            'Agent CLI': (EVALUATOR_DIR / 'AGENT_CLI.md').read_text(encoding='utf-8'),
            'CANNBot README': (EVALUATOR_DIR.parent / 'modes/cannbot/README.md').read_text(encoding='utf-8'),
            'CANNBot design': (EVALUATOR_DIR.parent / 'modes/cannbot/DESIGN.md').read_text(encoding='utf-8'),
        }
        for name, document in documents.items():
            with self.subTest(document=name):
                text = ' '.join(document.split())
                self.assertIn('ptobench evaluate', text)
                self.assertRegex(text, r'(?i)Evaluate (?:budget|allowance).*?1.*?accepted')
                self.assertRegex(text, r'(?i)Profile.*?(?:cost is 0|cost 0|allowance cost is 0|额度成本为 0|0; recorded)')
                self.assertRegex(text, r'(?i)Profile.*?recorded as a hardware action')
                self.assertRegex(text, r'(?i)Fork.*?(?:cost is 0|cost 0|0; recorded)')
                self.assertRegex(text, r'(?i)(?:failed or interrupted accepted Evaluate.*?consum|failed.*?interrupted.*?Evaluate.*?consum)')
                self.assertRegex(text, r'(?i)(?:Profile.*?available after.*?exhausted|Profile/Fork remain available)')
                self.assertRegex(text, r'(?i)(?:extra Evaluate requests are rejected by the CLI|CLI rejects new Evaluate)')
                self.assertNotIn('Each accepted evaluation or enabled diagnostic operation uses one', document)
                self.assertNotIn('diagnostics have no free NPU execution', document)
        for private_detail in ('warmup', 'measured launches', 'final verification', 'Snapshot',
                               'finish budget', 'protection/', 'staging/', 'raw_wiki'):
            self.assertNotIn(private_detail, guide)
        self.assertFalse(any('\u4e00' <= character <= '\u9fff' for character in guide))

    def test_workspace_omits_fixture_and_profile_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "case"
            sealed = root / "sealed"
            destination = root / "workspace"
            seed = case_dir / "public/candidate_seed"
            seed.mkdir(parents=True)
            sealed.mkdir()
            inputs = sealed / "fixtures/fixture_000/in"
            inputs.mkdir(parents=True)
            (inputs / "x.pt").write_bytes(b"fixed test input")
            (seed / "candidate.py").write_text("def entry():\n    pass\n", encoding="utf-8")
            (case_dir / "case.json").write_text("{}\n", encoding="utf-8")
            (case_dir / "public/config.py").write_text("VALUE = 1\n", encoding="utf-8")
            (case_dir / "public/comparator.json").write_text("{}\n", encoding="utf-8")
            (sealed / "certification.json").write_text(
                '{"status": "ready"}\n', encoding="utf-8",
            )
            case = {
                "case_id": "test.case",
                "fixture_ids": ["fixture_000"],
                "workspace": "public/candidate_seed",
                "module": "candidate.py",
                "entry": "entry",
                "config": "public/config.py",
                "comparator": "public/comparator.json",
                "environment": "a2a3_single_card",
            }

            with mock.patch.object(workspace.profile_modes, "enabled_modes", return_value={}), \
                 mock.patch.object(common, "resolve_case", return_value=(case_dir, sealed, case)):
                workspace.init_workspace(case_dir, sealed, case, destination)

            context = destination / ".benchmark_context"
            self.assertFalse((context / "mode.json").exists())
            self.assertFalse((context / "fixture_catalog.json").exists())
            self.assertFalse((context / "profiling_modes.json").exists())
            self.assertNotIn("| `ptobench profile", (context / "README.md").read_text().lower())
            self.assertTrue((destination / "candidate" / "candidate.py").is_file())
            self.assertTrue((destination / ".benchmark_context").is_dir())
            self.assertTrue((destination / ".benchmark_outputs").is_dir())
            self.assertFalse((destination / "candidate").is_symlink())
            self.assertFalse((destination / ".benchmark_context").is_symlink())
            self.assertFalse((destination / ".benchmark_outputs").is_symlink())
            share_gid = grp.getgrnam("ptobench-share").gr_gid
            expected_modes = {
                destination: 0o2750,
                destination / "candidate": 0o2770,
                destination / ".benchmark_context": 0o750,
                destination / ".benchmark_outputs": 0o750,
                destination / "candidate/candidate.py": 0o660,
                destination / "candidate/tuning.json": 0o660,
                destination / ".benchmark_context/README.md": 0o640,
                destination / ".benchmark_context/starting_point/candidate/candidate.py": 0o640,
            }
            for path, mode in expected_modes.items():
                metadata = path.lstat()
                self.assertEqual(metadata.st_uid, os.getuid(), path)
                self.assertEqual(metadata.st_gid, share_gid, path)
                self.assertEqual(metadata.st_mode & 0o7777, mode, path)
            self.assertEqual(
                (context / "starting_point" / "candidate" / "candidate.py").read_text(encoding="utf-8"),
                "def entry():\n    pass\n",
            )
            self.assertEqual(
                common.read_json(destination / "candidate" / "tuning.json"),
                {"memory_planner": "PYPTO", "analyze_auto_scopes_for_deps": False},
            )

    def test_init_rejects_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "workspace"
            destination.mkdir()
            (destination / "existing.py").write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit):
                workspace.init_workspace(Path("case"), Path("sealed"), {}, destination)

    def test_fork_copies_candidate_and_regenerates_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "case"
            sealed = root / "sealed"
            parent = root / "parent"
            child = root / "child"
            seed = case_dir / "public/candidate_seed"
            seed.mkdir(parents=True)
            sealed.mkdir()
            inputs = sealed / "fixtures/fixture_000/in"
            inputs.mkdir(parents=True)
            (inputs / "x.pt").write_bytes(b"fixed test input")
            (seed / "candidate.py").write_text("VALUE = 1\n", encoding="utf-8")
            (case_dir / "case.json").write_text("{}\n", encoding="utf-8")
            (case_dir / "public/config.py").write_text("VALUE = 1\n", encoding="utf-8")
            (case_dir / "public/comparator.json").write_text("{}\n", encoding="utf-8")
            (sealed / "certification.json").write_text(
                '{"status": "ready"}\n', encoding="utf-8",
            )
            case = {
                "case_id": "test.case",
                "fixture_ids": ["fixture_000"],
                "workspace": "public/candidate_seed",
                "module": "candidate.py",
                "entry": "entry",
                "config": "public/config.py",
                "comparator": "public/comparator.json",
                "environment": "a2a3_single_card",
            }
            with mock.patch.object(workspace.profile_modes, "enabled_modes", return_value={}), \
                 mock.patch.object(common, "resolve_case", return_value=(case_dir, sealed, case)):
                workspace.init_workspace(case_dir, sealed, case, parent)
                (parent / "candidate" / "candidate.py").write_text("VALUE = 2\n", encoding="utf-8")
                (parent / ".benchmark_outputs" / "latest_result.json").write_text("{}\n")
                result = workspace.fork_workspace(
                    parent, child,
                )

            self.assertEqual(result["parent_workspace"], str(parent.resolve()))
            self.assertEqual(
                (child / "candidate" / "candidate.py").read_text(encoding="utf-8"), "VALUE = 2\n",
            )
            self.assertEqual(
                (child / ".benchmark_context" / "starting_point" / "candidate" / "candidate.py").read_text(
                    encoding="utf-8"
                ),
                "VALUE = 2\n",
            )
            self.assertTrue((child / ".benchmark_outputs").is_dir())
            (child / "candidate" / "candidate.py").write_text("VALUE = 3\n", encoding="utf-8")
            self.assertEqual((parent / "candidate" / "candidate.py").read_text(encoding="utf-8"), "VALUE = 2\n")
            binding = common.read_json(child / ".benchmark_context" / "workspace.json")
            self.assertEqual(binding["parent_workspace"], str(parent.resolve()))
            self.assertEqual(binding["workspace"], str(child.resolve()))

    def test_context_recovery_preserves_initial_source_and_current_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir, sealed, candidate = root / "case", root / "sealed", root / "workspace"
            seed = case_dir / "public/candidate_seed"
            seed.mkdir(parents=True)
            sealed.mkdir()
            inputs = sealed / "fixtures/fixture_000/in"
            inputs.mkdir(parents=True)
            (inputs / "x.pt").write_bytes(b"fixed test input")
            (seed / "candidate.py").write_text("VERSION = 1\n")
            (seed / "helper.py").write_text("VALUE = 'initial'\n")
            for path in (case_dir / "case.json", case_dir / "public/comparator.json"):
                path.write_text("{}\n")
            (case_dir / "public/config.py").write_text("WIDTH = 2\n")
            (sealed / "certification.json").write_text('{"status":"ready"}')
            case = {"case_id": "fake.case", "fixture_ids": ["fixture_000"], "workspace": "public/candidate_seed",
                    "module": "candidate.py", "entry": "entry", "config": "public/config.py",
                    "comparator": "public/comparator.json", "environment": "fake"}
            with mock.patch.object(workspace, "_environment_summary", return_value={"profile": "original"}), \
                    mock.patch.object(workspace.profile_modes, "enabled_modes", return_value={}):
                workspace.init_workspace(case_dir, sealed, case, candidate)
            context = candidate / ".benchmark_context"
            protection = root / "protection"
            integrity.capture_protection(protection, {
                "context": integrity.ProtectionRoot(context, "context", readonly=False),
            })
            (candidate / "candidate" / "candidate.py").write_text("VERSION = 2\n")
            (candidate / "candidate" / "helper.py").write_text("VALUE = 'optimized'\n")
            (case_dir / "public/config.py").write_text("WIDTH = 999\n")
            original_candidate = workspace.source_manifest(candidate)
            initial = context / "starting_point/candidate/candidate.py"
            initial.chmod(0o644)
            initial.write_text("fake initial\n")
            (context / "config.py").unlink()
            shutil.rmtree(context / "starting_point")
            (context / "starting_point").write_text("wrong type")
            (context / "extra").write_text("rogue")
            report = integrity.check_and_restore(protection, "context")
            self.assertTrue(report.restored_paths)
            self.assertEqual(initial.read_text(), "VERSION = 1\n")
            self.assertEqual((context / "starting_point/candidate/helper.py").read_text(), "VALUE = 'initial'\n")
            self.assertEqual(workspace.source_manifest(candidate), original_candidate)
            self.assertEqual((context / "config.py").read_text(), "WIDTH = 2\n")
            self.assertEqual((context / "environment.json").read_text(), '{\n  "profile": "original"\n}\n')
            self.assertEqual(initial.stat().st_mode & 0o777, 0o640)
            self.assertFalse((context / "extra").exists())


class CandidateCopyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.target = self.root / "source", self.root / "target"
        (self.source / "helpers/deep/empty").mkdir(parents=True)
        (self.source / "candidate.py").write_text("import helpers.kernel\n")
        (self.source / "helpers/kernel.py").write_text("VERSION = 1\n")

    def test_complete_independent_copy_excludes_outputs_and_caches_recursively(self):
        for parent in (self.source, self.source / "helpers"):
            for name in workspace.CANDIDATE_EXCLUDES:
                directory = parent / name
                directory.mkdir()
                (directory / "ignored").write_text("ignored")
            for name in ("ignored.pyc", "ignored.pyo"):
                (parent / name).write_bytes(b"cached")
        manifest = workspace.copy_candidate_tree(self.source, self.target)
        self.assertEqual(manifest, workspace.source_manifest(self.source))
        self.assertEqual(manifest, workspace.source_manifest(self.target))
        self.assertEqual([item["path"] for item in manifest["files"]],
                         ["candidate.py", "helpers/kernel.py"])
        self.assertTrue((self.target / "helpers/deep/empty").is_dir())
        self.assertNotEqual((self.target / "candidate.py").stat().st_ino,
                            (self.source / "candidate.py").stat().st_ino)
        for excluded in workspace.CANDIDATE_EXCLUDES:
            self.assertFalse((self.target / excluded).exists())
            self.assertFalse((self.target / "helpers" / excluded).exists())

    def test_hash_tracks_names_and_bytes_but_not_cached_outputs(self):
        original = workspace.source_manifest(self.source)["content_sha256"]
        (self.source / ".benchmark_outputs").mkdir()
        (self.source / ".benchmark_outputs/result.json").write_text("new result")
        (self.source / "cache.pyc").write_bytes(b"cache")
        self.assertEqual(workspace.source_manifest(self.source)["content_sha256"], original)
        (self.source / "helpers/kernel.py").rename(self.source / "helpers/renamed.py")
        renamed = workspace.source_manifest(self.source)["content_sha256"]
        self.assertNotEqual(original, renamed)
        (self.source / "helpers/renamed.py").write_text("VERSION = 2\n")
        self.assertNotEqual(renamed, workspace.source_manifest(self.source)["content_sha256"])

    def test_non_candidate_files_are_rejected_but_generic_artifacts_can_be_copied(self):
        extra = self.source / 'notes.md'
        extra.write_text('not Candidate source')
        with self.assertRaisesRegex(workspace.InvalidWorkspaceError, 'outside the Candidate'):
            workspace.copy_candidate_tree(self.source, self.target)
        self.assertFalse(self.target.exists())
        extra.unlink()
        artifact = self.root / 'artifact'
        artifact.mkdir()
        (artifact / 'trace.json').write_text('{}\n')
        (artifact / 'trace.bin').write_bytes(b'\\x00\\xff')
        workspace.copy_tree(artifact, self.target)
        self.assertEqual((self.target / 'trace.json').read_text(), '{}\n')
        self.assertEqual((self.target / 'trace.bin').read_bytes(), b'\\x00\\xff')

    def test_source_links_and_fifos_fail_without_publishing_or_reading_outside(self):
        sentinel = self.root / "sentinel"
        sentinel.write_text("outside")
        for name, directory in (("linked.py", False), ("linked_dir", True), ("__pycache__", False)):
            with self.subTest(name=name):
                node = self.source / name
                node.symlink_to(sentinel if not directory else self.root, target_is_directory=directory)
                with self.assertRaises(workspace.InvalidWorkspaceError) as failure:
                    workspace.copy_candidate_tree(self.source, self.target)
                self.assertEqual(failure.exception.error_code, "invalid_workspace")
                node.unlink()
                self.assertFalse(self.target.exists())
                self.assertEqual(sentinel.read_text(), "outside")
        fifo = self.source / "pipe"
        os.mkfifo(fifo)
        with self.assertRaises(workspace.InvalidWorkspaceError):
            workspace.copy_candidate_tree(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.glob(".target.copy_*")))

    def test_source_root_and_ancestor_links_are_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        for source in (alias, alias / "helpers"):
            with self.subTest(source=source), self.assertRaises(workspace.InvalidWorkspaceError):
                workspace.copy_candidate_tree(source, self.target)
        self.assertFalse(self.target.exists())

    def test_link_substituted_after_scan_is_rejected_without_publishing(self):
        sentinel = self.root / "sentinel"
        sentinel.write_text("outside")
        actual_copy = workspace._copy_source_file

        def swap_during_copy(source, target, entry):
            source.unlink()
            source.symlink_to(sentinel)
            return actual_copy(source, target, entry)

        with mock.patch.object(workspace, "_copy_source_file", side_effect=swap_during_copy):
            with self.assertRaises(workspace.InvalidWorkspaceError):
                workspace.copy_candidate_tree(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(sentinel.read_text(), "outside")
        self.assertFalse(list(self.root.glob(".target.copy_*")))

    def test_target_links_and_nonempty_target_never_overwrite_existing_files(self):
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_text("outside")
        self.target.symlink_to(outside, target_is_directory=True)
        for destination in (self.target, self.target / "child"):
            with self.subTest(destination=destination), self.assertRaises(workspace.InvalidWorkspaceError):
                workspace.copy_candidate_tree(self.source, destination)
        self.assertEqual(sentinel.read_text(), "outside")
        self.assertFalse((outside / "child").exists())
        self.target.unlink()
        self.target.mkdir()
        (self.target / "existing.py").write_text("existing")
        with self.assertRaises(workspace.InvalidWorkspaceError):
            workspace.copy_candidate_tree(self.source, self.target)
        self.assertEqual((self.target / "existing.py").read_text(), "existing")
        self.assertFalse((self.target / "candidate.py").exists())

    def test_parallel_edit_retries_and_publishes_one_consistent_tree(self):
        copying, edited = threading.Event(), threading.Event()
        actual_copy = workspace._copy_source_file
        first = True
        errors = []

        def edit():
            try:
                if not copying.wait(3):
                    raise AssertionError("copy was not reached")
                (self.source / "helpers/kernel.py").write_text("VERSION = 2\n")
            except BaseException as error:
                errors.append(error)
            finally:
                edited.set()

        def synchronized_copy(*args):
            nonlocal first
            if first:
                first = False
                copying.set()
                self.assertTrue(edited.wait(3))
            return actual_copy(*args)

        worker = threading.Thread(target=edit)
        worker.start()
        try:
            with mock.patch.object(workspace, "_copy_source_file", side_effect=synchronized_copy):
                manifest = workspace.copy_candidate_tree(self.source, self.target)
        finally:
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual((self.target / "helpers/kernel.py").read_text(), "VERSION = 2\n")
        self.assertEqual(manifest, workspace.source_manifest(self.source))
        self.assertEqual(manifest, workspace.source_manifest(self.target))

    def test_continuously_changing_source_fails_after_three_retries_without_partial_publish(self):
        actual_copy = workspace._copy_source_tree
        attempts = []
        self.target.mkdir()

        def change_after_copy(*args):
            actual_copy(*args)
            attempts.append(1)
            with (self.source / "candidate.py").open("a") as stream:
                stream.write("# changed\n")

        with mock.patch.object(workspace, "_copy_source_tree", side_effect=change_after_copy):
            with self.assertRaisesRegex(workspace.InvalidWorkspaceError, "stably"):
                workspace.copy_candidate_tree(self.source, self.target)
        self.assertEqual(len(attempts), 4, "one attempt plus at most three retries")
        self.assertFalse(list(self.target.iterdir()))
        self.assertFalse(list(self.root.glob(".target.copy_*")))

    def test_source_and_destination_overlap_is_rejected(self):
        for target in (self.source, self.source / "child", self.root):
            with self.subTest(target=target), self.assertRaises(workspace.InvalidWorkspaceError):
                workspace.copy_candidate_tree(self.source, target)
        self.assertEqual((self.source / "candidate.py").read_text(), "import helpers.kernel\n")

    def test_non_utf8_filename_is_rejected_before_publishing(self):
        filename = os.fsencode(self.source) + b"/invalid_\xff.py"
        with open(filename, "wb") as stream:
            stream.write(b"source")
        with self.assertRaises(workspace.InvalidWorkspaceError):
            workspace.copy_candidate_tree(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.glob(".target.copy_*")))


if __name__ == "__main__":
    unittest.main()
