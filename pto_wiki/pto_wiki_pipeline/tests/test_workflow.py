import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pto_wiki.pto_wiki_pipeline import workflow
from pto_wiki.pto_wiki_pipeline.workflow import (
    WorkflowError,
    read_env,
    validate_candidate,
    validate_manifest,
    validate_mapping,
    validate_source_map,
)


class WorkflowTests(unittest.TestCase):
    def test_manifest_hash_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "doc.md"
            source.write_text("first", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "input_files": [{
                    "path": "doc.md",
                    "bytes": 5,
                    "sha256": hashlib.sha256(b"first").hexdigest(),
                }]
            }), encoding="utf-8")
            validate_manifest(manifest, root)
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "differs from manifest"):
                validate_manifest(manifest, root)

    def test_env_parser_keeps_values_literal(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "PTO_WIKI_AGENT=codex\n"
                "PTO_WIKI_MODEL=gpt-6.1-sol\n"
                "PTO_WIKI_REASONING=high\n"
                "PTO_WIKI_BASE_URL=https://api.aixuanhub.me/v1\n"
                "PTO_WIKI_API_KEY=$(touch marker)\n",
                encoding="utf-8",
            )
            values = read_env(env_file)
            self.assertEqual(values["PTO_WIKI_API_KEY"], "$(touch marker)")
            self.assertFalse((Path(directory) / "marker").exists())

    def test_mapping_must_cover_each_manifest_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            manifest = {"input_files": [{"path": "a.md"}, {"path": "b.md"}]}
            digest = "a" * 64
            path.write_text(json.dumps({
                "schema_version": 1,
                "manifest_sha256": digest,
                "topics": [{"id": "topic", "title": "Topic", "sources": ["a.md"]}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "coverage mismatch"):
                validate_mapping(path, manifest, digest)

    def test_publish_keeps_previous_knowledge_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "pypto_wiki"
            old.mkdir()
            (old / "README.md").write_text("old", encoding="utf-8")
            pipeline = root / "pipeline"
            run_dir = pipeline / ".work" / "run-1"
            candidate = run_dir / "candidate_pypto_wiki"
            candidate.mkdir(parents=True)
            (candidate / "README.md").write_text("new", encoding="utf-8")
            for name in ("mapping.json", "source-map.json", "review.json"):
                (run_dir / name).write_text("{}", encoding="utf-8")
            provenance = pipeline / "provenance"
            with patch.object(workflow, "PYPTO_WIKI", old), patch.object(workflow, "PROVENANCE", provenance):
                result = workflow.publish(run_dir, candidate, {"run_id": "run-1"})
            self.assertEqual((old / "README.md").read_text(encoding="utf-8"), "new")
            self.assertEqual((run_dir / "previous_pypto_wiki" / "README.md").read_text(encoding="utf-8"), "old")
            self.assertTrue((result / "source-map.json").is_file())

    def test_candidate_links_must_resolve_inside_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate"
            topic = candidate / "topics" / "api"
            topic.mkdir(parents=True)
            (candidate / "README.md").write_text("[Index](INDEX.md)", encoding="utf-8")
            (candidate / "INDEX.md").write_text("[API](topics/api/README.md)", encoding="utf-8")
            (topic / "README.md").write_text("Topic", encoding="utf-8")
            validate_candidate(candidate)
            (candidate / "INDEX.md").write_text("[API](../outside.md)", encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "Broken or external"):
                validate_candidate(candidate)

    def test_candidate_ignores_link_syntax_inside_fenced_code(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate"
            topic = candidate / "topics" / "api"
            topic.mkdir(parents=True)
            (candidate / "README.md").write_text("Wiki", encoding="utf-8")
            (candidate / "INDEX.md").write_text(
                "`[](const auto& kwargs)`\n\n````cpp\n[](const auto& kwargs)\n````\n",
                encoding="utf-8",
            )
            (topic / "README.md").write_text("Topic", encoding="utf-8")
            validate_candidate(candidate)

    def test_source_map_rejects_paths_outside_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            (root / "outside.md").write_text("outside", encoding="utf-8")
            source_map = root / "source-map.json"
            source_map.write_text(json.dumps({
                "schema_version": 1,
                "manifest_sha256": "b" * 64,
                "sources": [{"path": "doc.md", "outputs": ["../outside.md"]}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "outside candidate"):
                validate_source_map(
                    source_map,
                    candidate,
                    {"input_files": [{"path": "doc.md"}]},
                    "b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
