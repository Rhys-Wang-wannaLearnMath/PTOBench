#!/usr/bin/env python3
"""Run the offline PTO wiki organization workflow with Codex CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit


PIPELINE = Path(__file__).resolve().parent
PTO_WIKI = PIPELINE.parent
CLEAN_WIKI = PTO_WIKI / "clean_wiki"
OPT_WIKI = PTO_WIKI / "opt_wiki"
MANIFEST = CLEAN_WIKI / "INPUT_MANIFEST.json"
WORK = PIPELINE / ".work"
PROVENANCE = PIPELINE / "provenance"
ENV_FILE = PIPELINE / ".env"
ENV_KEYS = {
    "PTO_WIKI_AGENT",
    "PTO_WIKI_MODEL",
    "PTO_WIKI_REASONING",
    "PTO_WIKI_BASE_URL",
    "PTO_WIKI_API_KEY",
}


class WorkflowError(Exception):
    pass


def read_env(path: Path = ENV_FILE) -> dict[str, str]:
    """Read literal KEY=value lines without shell execution or expansion."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkflowError(f"Cannot read {path}; copy .env.example to .env") from error
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or key not in ENV_KEYS or key in values or "\0" in value:
            raise WorkflowError(f"Invalid or duplicate setting in {path}: {key or '<empty>'}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    missing = ENV_KEYS - values.keys()
    if missing:
        raise WorkflowError(f"Missing .env settings: {', '.join(sorted(missing))}")
    return values


def validate_settings(settings: dict[str, str], require_key: bool) -> str | None:
    expected = {
        "PTO_WIKI_AGENT": "codex",
        "PTO_WIKI_MODEL": "gpt-6.1-sol",
        "PTO_WIKI_REASONING": "high",
        "PTO_WIKI_BASE_URL": "https://api.aixuanhub.me/v1",
    }
    for key, value in expected.items():
        if settings[key] != value:
            raise WorkflowError(f"{key} must be {value!r}")
    parsed = urlsplit(settings["PTO_WIKI_BASE_URL"])
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise WorkflowError("PTO_WIKI_BASE_URL must be a valid HTTPS URL")
    codex = shutil.which("codex")
    if not codex:
        raise WorkflowError("Codex CLI was not found on PATH")
    if require_key and not settings["PTO_WIKI_API_KEY"].strip():
        raise WorkflowError("Fill PTO_WIKI_API_KEY in pto_wiki_pipeline/.env before running")
    return codex


def validate_manifest(manifest_path: Path = MANIFEST, clean_root: Path = CLEAN_WIKI) -> tuple[dict, str]:
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"Cannot read input manifest: {error}") from error
    files = manifest.get("input_files")
    if not isinstance(files, list) or not files:
        raise WorkflowError("Input manifest has no input_files list")
    clean_root = clean_root.resolve()
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise WorkflowError("Input manifest contains an invalid file entry")
        relative = item.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise WorkflowError(f"Invalid or duplicate input path: {relative!r}")
        seen.add(relative)
        source = clean_root / relative
        try:
            resolved = source.resolve(strict=True)
            if not resolved.is_relative_to(clean_root) or not resolved.is_file():
                raise WorkflowError(f"Input path escapes clean_wiki or is not a file: {relative}")
            content = resolved.read_bytes()
        except OSError as error:
            raise WorkflowError(f"Missing manifest input {relative}: {error}") from error
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != item.get("bytes") or digest != item.get("sha256"):
            raise WorkflowError(f"Input differs from manifest: {relative}")
    return manifest, hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"Expected valid JSON at {path}: {error}") from error
    if not isinstance(data, dict):
        raise WorkflowError(f"Expected a JSON object at {path}")
    return data


def validate_mapping(path: Path, manifest: dict, manifest_hash: str) -> None:
    data = load_json(path)
    if data.get("schema_version") != 1 or data.get("manifest_sha256") != manifest_hash:
        raise WorkflowError("Mapping schema or input manifest hash is incorrect")
    expected = {item["path"] for item in manifest["input_files"]}
    found: set[str] = set()
    topic_ids: set[str] = set()
    topics = data.get("topics")
    if not isinstance(topics, list) or not topics:
        raise WorkflowError("Mapping has no topics")
    for topic in topics:
        if not isinstance(topic, dict) or not isinstance(topic.get("id"), str) or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", topic["id"]
        ):
            raise WorkflowError("Mapping contains an invalid topic")
        if topic["id"] in topic_ids:
            raise WorkflowError(f"Duplicate topic id: {topic['id']}")
        topic_ids.add(topic["id"])
        sources = topic.get("sources")
        if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
            raise WorkflowError(f"Topic {topic['id']} has no source list")
        found.update(sources)
    unknown = found - expected
    missing = expected - found
    if unknown or missing:
        raise WorkflowError(f"Mapping coverage mismatch: {len(missing)} missing, {len(unknown)} unknown")


def validate_source_map(path: Path, candidate: Path, manifest: dict, manifest_hash: str) -> None:
    data = load_json(path)
    if data.get("schema_version") != 1 or data.get("manifest_sha256") != manifest_hash:
        raise WorkflowError("Source map schema or input manifest hash is incorrect")
    expected = {item["path"] for item in manifest["input_files"]}
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise WorkflowError("Source map has no sources list")
    found: set[str] = set()
    for item in sources:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise WorkflowError("Source map contains an invalid source entry")
        source = item["path"]
        if source in found:
            raise WorkflowError(f"Duplicate source map entry: {source}")
        found.add(source)
        outputs = item.get("outputs")
        if not isinstance(outputs, list) or not outputs or any(not isinstance(output, str) for output in outputs):
            raise WorkflowError(f"Source has no output mapping: {source}")
        for output in outputs:
            if Path(output).is_absolute():
                raise WorkflowError(f"Invalid output path for {source}")
            target = (candidate / output).resolve()
            if not target.is_relative_to(candidate.resolve()) or not target.is_file():
                raise WorkflowError(f"Mapped output is missing or outside candidate: {output}")
    if found != expected:
        raise WorkflowError(f"Source map coverage mismatch: {len(expected - found)} missing, {len(found - expected)} unknown")


def validate_candidate(candidate: Path) -> None:
    for path in candidate.rglob("*"):
        if path.is_symlink():
            raise WorkflowError(f"Candidate contains a symbolic link: {path.relative_to(candidate)}")
    if not (candidate / "README.md").is_file() or not (candidate / "INDEX.md").is_file():
        raise WorkflowError("Candidate must contain README.md and INDEX.md")
    topics = candidate / "topics"
    if not topics.is_dir() or not any(topics.rglob("*.md")):
        raise WorkflowError("Candidate has no topic documents")
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for markdown in candidate.rglob("*.md"):
        try:
            text = markdown.read_text(encoding="utf-8")
        except UnicodeError as error:
            raise WorkflowError(f"Candidate Markdown is not UTF-8: {markdown.relative_to(candidate)}") from error
        visible_lines = []
        fence_char = None
        fence_length = 0
        for line in text.splitlines():
            fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if fence_char is None:
                if fence:
                    fence_char = fence.group(1)[0]
                    fence_length = len(fence.group(1))
                    visible_lines.append("")
                else:
                    visible_lines.append(line)
                continue
            visible_lines.append("")
            if re.fullmatch(rf" {{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*", line):
                fence_char = None
                fence_length = 0
        text = re.sub(r"(`+).*?\1", "", "\n".join(visible_lines), flags=re.DOTALL)
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if "clean_wiki" in target:
                raise WorkflowError(f"Candidate links back to clean_wiki: {markdown.relative_to(candidate)}")
            target_path = unquote(target.split("#", 1)[0])
            resolved = (markdown.parent / target_path).resolve()
            if not resolved.is_relative_to(candidate.resolve()) or not resolved.exists():
                raise WorkflowError(f"Broken or external candidate link in {markdown.relative_to(candidate)}: {target}")


def render_prompt(name: str, **values: str) -> str:
    prompt = (PIPELINE / "prompts" / name).read_text(encoding="utf-8")
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", value)
    return prompt


def codex_command(codex: str, cwd: Path, sandbox: str, settings: dict[str, str], output: Path) -> list[str]:
    base_url = settings["PTO_WIKI_BASE_URL"]
    return [
        codex,
        "--cd", str(cwd),
        "--sandbox", sandbox,
        "--ask-for-approval", "never",
        "--model", settings["PTO_WIKI_MODEL"],
        "--config", 'model_provider="pto_wiki"',
        "--config", 'model_providers.pto_wiki.name="PTO Wiki"',
        "--config", f'model_providers.pto_wiki.base_url="{base_url}"',
        "--config", 'model_providers.pto_wiki.env_key="PTO_WIKI_API_KEY"',
        "--config", 'model_providers.pto_wiki.wire_api="responses"',
        "--config", f'model_reasoning_effort="{settings["PTO_WIKI_REASONING"]}"',
        "--config", "sandbox_workspace_write.network_access=false",
        "exec",
        "--ignore-user-config",
        "--output-last-message", str(output),
    ]


def run_codex(codex: str, cwd: Path, sandbox: str, settings: dict[str, str], output: Path, prompt: str) -> None:
    child_env = os.environ.copy()
    child_env["PTO_WIKI_API_KEY"] = settings["PTO_WIKI_API_KEY"]
    result = subprocess.run(
        codex_command(codex, cwd, sandbox, settings, output),
        input=prompt,
        text=True,
        env=child_env,
        check=False,
    )
    if result.returncode:
        raise WorkflowError(f"Codex exited with status {result.returncode}; work files are retained")
    if not output.is_file() or not output.stat().st_size:
        raise WorkflowError(f"Codex did not produce the expected result: {output}")


def publish(run_dir: Path, candidate: Path, run_metadata: dict) -> Path:
    if not candidate.is_dir() or not OPT_WIKI.is_dir():
        raise WorkflowError("Candidate and current opt_wiki must both be directories")
    provenance_dir = PROVENANCE / run_dir.name
    provenance_dir.mkdir(parents=True, exist_ok=False)
    run_metadata["review_status"] = "pass"
    for name in ("mapping.json", "source-map.json", "review.json"):
        shutil.copy2(run_dir / name, provenance_dir / name)
    metadata_path = run_dir / "run.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(metadata_path, provenance_dir / "run.json")

    backup = run_dir / "previous_opt_wiki"
    if OPT_WIKI.exists():
        if backup.exists():
            raise WorkflowError(f"Backup path already exists: {backup}")
        os.replace(OPT_WIKI, backup)
    try:
        os.replace(candidate, OPT_WIKI)
    except OSError:
        if backup.exists() and not OPT_WIKI.exists():
            os.replace(backup, OPT_WIKI)
        raise
    return provenance_dir


def check() -> int:
    settings = read_env()
    codex = validate_settings(settings, require_key=False)
    manifest, digest = validate_manifest()
    print(f"Input manifest: valid ({len(manifest['input_files'])} files, sha256 {digest})")
    print(f"Codex CLI: {codex}")
    print(f"Model: {settings['PTO_WIKI_MODEL']} / reasoning {settings['PTO_WIKI_REASONING']}")
    print(f"Base URL: {settings['PTO_WIKI_BASE_URL']}")
    print(f"API key: {'configured' if settings['PTO_WIKI_API_KEY'].strip() else 'empty; fill .env before run'}")
    return 0


def run() -> int:
    settings = read_env()
    codex = validate_settings(settings, require_key=True)
    manifest, manifest_hash = validate_manifest()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = WORK / run_id
    candidate = run_dir / "candidate_opt_wiki"
    candidate.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "manifest_sha256": manifest_hash,
        "input_count": len(manifest["input_files"]),
        "agent": settings["PTO_WIKI_AGENT"],
        "model": settings["PTO_WIKI_MODEL"],
        "reasoning": settings["PTO_WIKI_REASONING"],
        "base_url": settings["PTO_WIKI_BASE_URL"],
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    common = {"CLEAN_WIKI": str(CLEAN_WIKI), "MANIFEST_SHA256": manifest_hash}
    print("[1/3] Mapping source documents")
    run_codex(
        codex, run_dir, "read-only", settings, run_dir / "mapping.json",
        render_prompt("01-map.md", **common),
    )
    validate_manifest()
    validate_mapping(run_dir / "mapping.json", manifest, manifest_hash)

    print("[2/3] Organizing opt_wiki")
    run_codex(
        codex, candidate, "workspace-write", settings, run_dir / "organize-result.md",
        render_prompt(
            "02-organize.md",
            **common,
            MAPPING=str(run_dir / "mapping.json"),
            CANDIDATE=str(candidate),
            REVIEW_FEEDBACK="",
        ),
    )
    validate_manifest()
    source_map = candidate / "source-map.json"
    validate_source_map(source_map, candidate, manifest, manifest_hash)
    validate_candidate(candidate)

    print("[3/3] Independent review")
    passed = False
    feedback = ""
    for review_number in range(1, 4):
        review_path = run_dir / f"review-{review_number}.json"
        run_codex(
            codex, run_dir, "read-only", settings, review_path,
            render_prompt(
                "03-review.md",
                CLEAN_WIKI=str(CLEAN_WIKI),
                CANDIDATE=str(candidate),
                MAPPING=str(run_dir / "mapping.json"),
                SOURCE_MAP=str(source_map),
            ),
        )
        review = load_json(review_path)
        if not isinstance(review.get("findings"), list):
            raise WorkflowError(f"Invalid review response; see {review_path}")
        if review.get("status") == "pass" and not review["findings"]:
            shutil.copy2(review_path, run_dir / "review.json")
            passed = True
            break
        if review.get("status") != "revise" or not review["findings"]:
            raise WorkflowError(f"Invalid or incomplete review; see {review_path}")
        feedback = json.dumps(review["findings"], ensure_ascii=False, indent=2)
        if review_number < 3:
            run_codex(
                codex, candidate, "workspace-write", settings,
                run_dir / f"organize-revision-{review_number}.md",
                render_prompt(
                    "02-organize.md",
                    **common,
                    MAPPING=str(run_dir / "mapping.json"),
                    CANDIDATE=str(candidate),
                    REVIEW_FEEDBACK=feedback,
                ),
            )
            validate_manifest()
            validate_source_map(source_map, candidate, manifest, manifest_hash)
            validate_candidate(candidate)
    if not passed:
        raise WorkflowError(f"Review still requires revision after two attempts; see {run_dir}")

    validate_manifest()
    shutil.copy2(source_map, run_dir / "source-map.json")
    source_map.unlink()
    provenance_dir = publish(run_dir, candidate, metadata)
    print(f"Published opt_wiki: {OPT_WIKI}")
    print(f"Review and source records: {provenance_dir}")
    print(f"Previous opt_wiki backup: {run_dir / 'previous_opt_wiki'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    args = parser.parse_args()
    try:
        return check() if args.command == "check" else run()
    except WorkflowError as error:
        print(f"Workflow error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
