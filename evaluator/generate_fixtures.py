#!/usr/bin/env python3
"""Materialize Case fixtures through the frozen official TensorSpec builder."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import torch
import common

def _save(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.detach().cpu().contiguous(), path)

def _materialize(case_dir: Path, sealed: Path, case: dict, fixture_id: str, seed: int) -> None:
    builder = common.load_module(sealed / "fixture_builder.py", f"fixture_builder_{sealed.name}")
    specs = builder.build(case_dir, sealed, case, seed)
    root = sealed / "fixtures" / fixture_id
    inputs, outputs = root / "in", root / "out"
    if root.exists():
        shutil.rmtree(root)
    inputs.mkdir(parents=True)
    outputs.mkdir()
    output_names = set(case["correctness"]["outputs"])
    state_names = set(case.get("state", {}).get("initial_state", []))
    for spec in specs:
        if hasattr(spec, "value") and not hasattr(spec, "shape"):
            _save(inputs / f"{spec.name}.pt", torch.as_tensor(spec.value))
            continue
        if spec.name in output_names and spec.name not in state_names:
            continue
        value = spec.create_tensor()
        _save(inputs / f"{spec.name}.pt", value)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "fixture_id": fixture_id,
        "source_id": f"public_normal_{seed}", "seed": seed,
        "generation": "official.build_tensor_specs",
        "inputs_dir": "in", "outputs_dir": "out",
        "scalars": list(case.get("scalars", [])),
    }, indent=2) + "\n", encoding="utf-8")

def generate_case(case_dir: Path, sealed: Path, case: dict) -> None:
    policy_file = sealed / "fixture_policy.json"
    slots = common.read_json(policy_file)["fixture_seeds"]
    if list(slots) != ["fixture_000"] or case.get("fixture_ids") != ["fixture_000"]:
        raise ValueError("v0.2 Cases must declare exactly one fixture: fixture_000")
    fixture_root = sealed / "fixtures"
    for old in fixture_root.iterdir() if fixture_root.exists() else ():
        if old.is_dir() and old.name not in slots:
            shutil.rmtree(old)
    for fixture_id, seed in slots.items():
        _materialize(case_dir, sealed, case, fixture_id, seed)
    common.write_json(sealed / "certification.json", {
        "schema_version": 1,
        "status": "pending",
        "case_id": case["case_id"],
        "ref_output_cache": "pending",
        "cpu_oracle_role": "one_time_ref_certification",
    })
    (sealed / "ref_performance.json").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    args = parser.parse_args()
    case_dir, sealed, case = common.resolve_case(args.case)
    generate_case(case_dir, sealed, case)
    print(f"{case['case_id']}: generated official fixtures")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
