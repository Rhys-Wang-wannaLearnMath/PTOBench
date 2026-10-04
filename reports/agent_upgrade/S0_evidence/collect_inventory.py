"""Read-only S0 inventory; does not load implementations or execute hardware.

Run from the repository root with .venv-official/bin/python. Asset checks here
are file-presence and recorded-certification checks, not fresh certification.
"""

import ast
import collections
import hashlib
import json
import math
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SELECTED = {
    "deepseek_v4.rmsnorm_decode_flash",
    "deepseek_v4.compressor_ratio4_flash",
    "deepseek_v4.decode_attention_hca_flash",
    "deepseek_v4.prefill_layer_ep2_hca_flash",
}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def files(directory):
    return [
        {"path": str(path.relative_to(BENCHMARK)), "size": path.stat().st_size}
        for path in sorted(directory.glob("*.pt"))
    ]


def collect():
    records = []
    for path in sorted((BENCHMARK / "cases").glob("*/*/case.json")):
        case = read(path)
        case_dir = path.parent
        sealed = BENCHMARK / "sealed_assets" / case_dir.relative_to(BENCHMARK / "cases")
        certification = read(sealed / "certification.json")
        policy_path = case_dir / case["comparator"]
        policy = read(policy_path)
        performance = read(sealed / "ref_performance.json")
        environment = read(BENCHMARK / "environments" / f"{case['environment']}.json")
        evaluator = sealed / "evaluate.py"
        factories = []
        if evaluator.is_file():
            factories = [
                {"name": node.name, "signature": ast.unparse(node.args)}
                for node in ast.parse(evaluator.read_text()).body
                if isinstance(node, ast.FunctionDef)
                and node.name in {"_compare", "_make_compare", "make_compare"}
            ]
        fixtures = []
        issues = []
        if not certification or certification.get("status") != "ready":
            issues.append("certification_not_ready")
        for label, asset in [
            ("evaluator", evaluator),
            ("candidate_seed_entry", case_dir / case["workspace"] / case["module"]),
            ("ref_entry", sealed / "ref/ref_npu.py"),
            ("oracle_entry", sealed / "oracle/ref_cpu.py"),
            ("comparator", policy_path),
            ("config", case_dir / case["config"]),
        ]:
            if not asset.is_file():
                issues.append(f"missing_{label}")
        comparator_hash = digest(policy_path)
        bound_hash = (certification or {}).get("comparator_sha256")
        if bound_hash != comparator_hash:
            issues.append("comparator_hash_unbound_or_mismatched")
        for fixture_id in case.get("fixture_ids", ["fixture_000"]):
            fixture = sealed / "fixtures" / fixture_id
            inputs = files(fixture / "in")
            outputs = files(fixture / "out")
            missing_outputs = [
                name for name in case["correctness"]["outputs"]
                if not (fixture / "out" / f"{name}.pt").is_file()
            ]
            latency = (performance or {}).get("fixtures", {}).get(fixture_id, {}).get("median_effective_us")
            valid_latency = type(latency) in (int, float) and math.isfinite(latency) and latency > 0
            if not (fixture / "manifest.json").is_file():
                issues.append(f"{fixture_id}:missing_manifest")
            if not inputs:
                issues.append(f"{fixture_id}:no_input_tensors")
            if missing_outputs:
                issues.append(f"{fixture_id}:missing_ref_outputs:{','.join(missing_outputs)}")
            if not valid_latency:
                issues.append(f"{fixture_id}:missing_or_invalid_ref_latency")
            fixtures.append({
                "fixture_id": fixture_id, "manifest": read(fixture / "manifest.json"),
                "inputs": inputs, "ref_outputs": outputs,
                "missing_declared_ref_outputs": missing_outputs,
                "ref_median_effective_us": latency,
                "ref_latency_valid": valid_latency,
            })
        runtime = {"device_id": (environment or {}).get("device", 0)}
        runtime.update((environment or {}).get("runtime", {}))
        runtime.update(case.get("runtime", {}))
        seed = case_dir / case["workspace"]
        records.append({
            "case_id": case["case_id"], "case_path": str(path.relative_to(BENCHMARK)),
            "certification": certification, "evaluator_exists": evaluator.is_file(),
            "comparator_factories": factories,
            "comparator_sha256": comparator_hash,
            "certification_comparator_matches": comparator_hash == bound_hash,
            "custom_outputs": {
                name: rule for name, rule in (policy or {}).get("outputs", {}).items()
                if rule.get("method") == "custom"
            },
            "environment": case["environment"], "environment_profile": environment,
            "runtime": runtime, "state": case.get("state"),
            "candidate_seed_sources": {
                str(source.relative_to(seed)): digest(source)
                for source in sorted(seed.rglob("*"))
                if source.is_file() and "__pycache__" not in source.parts
                and source.suffix not in {".pyc", ".pyo"}
            },
            "ref_sources": {
                str(source.relative_to(sealed / "ref")): digest(source)
                for source in sorted((sealed / "ref").rglob("*.py"))
            },
            "fixtures": fixtures, "asset_issues": issues,
        })
    counts = collections.Counter((record["certification"] or {}).get("status", "missing") for record in records)
    inventory = {
        "scope": "cases/*/*/case.json", "case_count": len(records),
        "certification_counts": dict(counts),
        "missing_evaluator_count": sum(not record["evaluator_exists"] for record in records),
        "checks": "Recorded certification, hashes, nonempty input directories, declared Ref outputs, positive Ref timing; no NPU execution or Fixture ABI certification.",
        "cases": records,
    }
    for filename, value in [
        ("inventory.json", inventory),
        ("selected_acceptance_cases.json", [record for record in records if record["case_id"] in SELECTED]),
    ]:
        (OUTPUT / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    for status in ("ready", "pending"):
        lines = [f"# S0 {status} inventory", "", "Presence checks only; recorded status is not fresh hardware acceptance.", "", "| Case | Evaluator | Asset issues |", "| --- | --- | --- |"]
        for record in records:
            if (record["certification"] or {}).get("status") == status:
                lines.append(f"| {record['case_id']} | {'present' if record['evaluator_exists'] else 'missing'} | {'; '.join(record['asset_issues']) or 'none detected'} |")
        (OUTPUT / f"{status}_cases.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in inventory.items() if key != "cases"}, ensure_ascii=False))
    for record in records:
        if record["case_id"] in SELECTED or (
            (record["certification"] or {}).get("status") == "ready" and record["asset_issues"]
        ):
            print(record["case_id"], "issues:", record["asset_issues"], "runtime:", record["runtime"])


if __name__ == "__main__":
    collect()
