#!/usr/bin/env python3
"""Discover SLUICE and execute the protected implementation frozen for this Run."""

import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluator")]


def main():
    target = ROOT / "modes/sluice/runtime.py"
    if len(sys.argv) == 3 and sys.argv[1] in ("run", "collect"):
        try:
            request = json.loads(Path(sys.argv[2]).read_text())
            method = Path(request["context"]["session_root"]) / "method"
            snapshot = method / "implementation"
            manifest = json.loads((method / "implementation.json").read_text())
            import file_tree

            actual = file_tree.scan_tree(snapshot, reject_special=True)
            files = [
                {"path": name, "sha256": "sha256:" + entry["sha256"]}
                for name, entry in sorted(actual.items())
                if entry["kind"] == "file"
            ]
            content_hash = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            )
            config = json.loads((method / "config.json").read_text())
            if (
                manifest.get("schema_version") != 1
                or manifest["files"] != files
                or manifest["content_sha256"] != content_hash
                or config["implementation_sha256"] != content_hash
            ):
                raise ValueError("Frozen implementation hash mismatch")
            sys.path.insert(0, str(snapshot))
            target = snapshot / "modes/sluice/runtime.py"
        except Exception:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "error",
                        "error_code": "infrastructure_error",
                        "message": "Frozen SLUICE implementation is unavailable or changed",
                    }
                )
            )
            return 5
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    raise SystemExit(main())
