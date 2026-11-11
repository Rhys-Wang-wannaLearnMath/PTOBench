"""Record the execution facts that define a PTOBench measurement."""
from __future__ import annotations

import hashlib
import json


POLICY_VERSION = "ptobench-execution-boundary-20261108-v3"


def policy(case: dict | None = None) -> dict:
    del case
    return {
        "version": POLICY_VERSION,
        "input_upload": "outside_execution_window",
        "initial_state_restore": "outside_execution_window",
        "final_validation_readback": "outside_execution_window",
        "required_execution_communication": "inside_execution_window",
    }


def execution_contract(case: dict, specs: list, runtime, input_identity: str) -> dict:
    """Bind actual compiled tensor ABI and fixture state to the measurement."""
    distributed = getattr(runtime, "distributed_config", None)
    body = {
        "timing_policy": policy(),
        "logical_workers": len(distributed.device_ids) if distributed else 1,
        "rank_layout": "case-defined",
        "fixture_input_identity_sha256": input_identity,
        "initial_state_names": sorted(case.get("state", {}).get("initial_state", [])),
        "restore_before_each_launch": case.get("state", {}).get("restore_before_each_launch", False),
        "tensors": [
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "dtype": str(spec.dtype),
                "resident": spec.resident if spec.resident not in (None, False) else None,
                "direction": getattr(spec, "direction", None),
            }
            for spec in specs if hasattr(spec, "shape")
        ],
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body["sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def validate_state_abi(case: dict, specs: list) -> None:
    """Ensure Case-declared reset state appears in the compiled tensor ABI."""
    names = set(case.get("state", {}).get("initial_state", []))
    compiled_names = {spec.name for spec in specs if hasattr(spec, "shape")}
    missing = names - compiled_names
    if missing:
        raise ValueError(f"mutable state absent from compiled Case ABI: {sorted(missing)}")


def attach(record: dict, case: dict) -> dict:
    record["timing_policy"] = policy()
    return record
