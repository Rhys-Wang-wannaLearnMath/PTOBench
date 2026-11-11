#!/usr/bin/env python3
"""PTOBench protocol entrypoint for the SLUICE single-session controller."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modes.sluice.foundation.protocol import ProtocolError, error_response, load_request, validate_top_level
from modes.sluice.engine import lifecycle
from modes._shared.config import ConfigError
from modes._shared.adapters import BackendUnavailable


def preflight(request):
    return lifecycle.preflight(request)


def prepare(request):
    return lifecycle.prepare(request)


def collect(request):
    return lifecycle.collect(request)


def run(request):
    return lifecycle.run(request)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] not in {"preflight", "prepare", "run", "collect"}:
        error = ProtocolError("usage: runner.py <preflight|prepare|run|collect> <request.json>")
        print(json.dumps(error_response(error), ensure_ascii=False, separators=(",", ":")))
        return 2
    operation, filename = args
    try:
        request = load_request(filename)
        request = validate_top_level(request, operation)
        result = {"preflight": preflight, "prepare": prepare, "run": run, "collect": collect}[operation](
            request
        )
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        return 1 if operation == "run" and result.get("status") == "failed" else 0
    except (ProtocolError, ConfigError, BackendUnavailable) as error:
        print(json.dumps(error_response(error), ensure_ascii=False, separators=(",", ":")))
        return 2 if error.error_code == "invalid_request" else 5
    except Exception:
        print(
            json.dumps(
                error_response(ProtocolError("SLUICE operation failed", "infrastructure_error")),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
