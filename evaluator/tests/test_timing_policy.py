"""Execution contracts follow Case declarations and compiled ABI only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EVALUATOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATOR_DIR))
import timing_policy


def test_policy_has_no_case_inventory_or_implicit_residency():
    plain = timing_policy.policy({"case_id": "family.case"})
    other = timing_policy.policy({"case_id": "profile_validation.case"})

    assert plain == other
    assert "inventory_sha256" not in plain
    assert "case_status" not in plain
    assert "resident_state" not in plain


def test_contract_preserves_case_residency_and_compiled_abi():
    case = {
        "state": {"initial_state": ["cache"], "restore_before_each_launch": True},
    }
    specs = [SimpleNamespace(name="cache", shape=(2, 4), dtype="float32",
                             resident="stacked", direction="inout")]
    runtime = SimpleNamespace(distributed_config=SimpleNamespace(device_ids=[0, 1]))

    contract = timing_policy.execution_contract(case, specs, runtime, "input-hash")
    body = {key: value for key, value in contract.items() if key != "sha256"}

    assert contract["logical_workers"] == 2
    assert contract["initial_state_names"] == ["cache"]
    assert contract["tensors"][0]["resident"] == "stacked"
    assert contract["tensors"][0]["direction"] == "inout"
    assert contract["sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_state_abi_checks_declared_state_presence_without_inventing_directions():
    case = {"state": {"initial_state": ["cache"]}}
    output = SimpleNamespace(name="cache", shape=(2,), direction="out")

    timing_policy.validate_state_abi(case, [output])
    with pytest.raises(ValueError, match="absent from compiled Case ABI"):
        timing_policy.validate_state_abi(case, [])
