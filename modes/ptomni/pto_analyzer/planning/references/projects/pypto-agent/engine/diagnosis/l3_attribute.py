"""L3 代码归因：包装 tune-swimlane/scripts/leafhash_to_code.py（子进程调用，按需执行）。"""

import shutil
import subprocess
from pathlib import Path


def _script_path():
    here = Path(__file__).resolve()
    skill_root = here.parents[4] / "ops" / "pypto-op-perf-tune" / "tune-swimlane" / "scripts"
    return skill_root / "leafhash_to_code.py"


def attribute(leaf_hashes, source_dir):
    script = _script_path()
    if not script.exists():
        return {
            "available": False,
            "reason": f"leafhash_to_code.py 不存在于 {script}",
        }
    cmd = ["python3", str(script)]
    for leaf in leaf_hashes:
        cmd += [str(leaf)]
    if source_dir:
        cmd += ["--src-dir", str(source_dir)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def run(artifacts_dir, diagnosis, source_dir=None):
    top_leafs = [item["leaf_hash"] for item in diagnosis.get("l2_localization", [])[:5]]
    if not top_leafs:
        diagnosis["layers_available"]["L3"] = "no_leaf_hashes"
        return diagnosis
    result = attribute(top_leafs, source_dir)
    diagnosis["layers_available"]["L3"] = result.get("available", False)
    diagnosis["l3_attribution"] = result if result.get("available") else []
    return diagnosis
