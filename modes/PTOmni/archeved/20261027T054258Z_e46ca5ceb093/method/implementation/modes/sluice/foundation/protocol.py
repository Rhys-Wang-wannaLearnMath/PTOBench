"""Validation and stable error handling for the independent method protocol."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping


class ProtocolError(ValueError):
    def __init__(self, message: str, error_code: str = "invalid_request"):
        super().__init__(message)
        if error_code not in {"invalid_request", "infrastructure_error"}:
            error_code = "infrastructure_error"
        self.error_code = error_code


_SENSITIVE = re.compile(r"(?:token|secret|password|passwd|api[_-]?key|credential|authorization|auth)", re.I)
_KNOWN_EXPERIMENT = {"mode", "case", "optimizer", "budget", "method_options", "output", "source"}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    return value


def _string(value: Any, name: str, *, pattern: str | None = None) -> str:
    if type(value) is not str or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    if pattern and not re.fullmatch(pattern, value):
        raise ProtocolError(f"{name} has an invalid format")
    return value


def _reject_sensitive(mapping: Mapping[str, Any], where: str) -> None:
    for key in mapping:
        if key in ("max_output_tokens", "maxTokens") and (type(mapping[key]) is int or mapping[key] == "unlimited"):
            continue
        if key == "supportsMaxOutputTokens" and type(mapping[key]) is bool:
            continue
        if key == "authentication" and mapping[key] in ("existing_host_configuration", "authenticated_host_configuration"):
            continue
        if not isinstance(key, str) or _SENSITIVE.search(key):
            raise ProtocolError(f"unsupported sensitive field in {where}")


def _reject_sensitive_deep(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        _reject_sensitive(value, where)
        for key, child in value.items():
            _reject_sensitive_deep(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive_deep(child, f"{where}[{index}]")


def load_request(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("request is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("request must be an object")
    return value


def validate_experiment(value: Any) -> dict[str, Any]:
    experiment = dict(_mapping(value, "experiment"))
    _reject_sensitive_deep(experiment, "experiment")
    unknown = set(experiment) - _KNOWN_EXPERIMENT
    if unknown:
        raise ProtocolError("experiment contains unsupported fields")
    if experiment.get("mode") != "sluice":
        raise ProtocolError("experiment.mode must be sluice")
    _string(experiment.get("case"), "experiment.case")
    optimizer = dict(_mapping(experiment.get("optimizer"), "experiment.optimizer"))
    _reject_sensitive_deep(optimizer, "optimizer")
    if set(optimizer) != {"id", "backend", "profile"}:
        raise ProtocolError("optimizer must contain id, backend and profile")
    _string(optimizer["id"], "optimizer.id", pattern=r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    _string(optimizer["backend"], "optimizer.backend", pattern=r"[a-z][a-z0-9_-]{0,31}")
    _string(optimizer["profile"], "optimizer.profile", pattern=r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    budget = dict(_mapping(experiment.get("budget"), "experiment.budget"))
    _reject_sensitive_deep(budget, "budget")
    if set(budget) - {"max_hardware_actions", "finish_actions"} or "max_hardware_actions" not in budget:
        raise ProtocolError("budget must contain public max_hardware_actions")
    maximum = budget["max_hardware_actions"]
    if maximum != "unlimited" and (type(maximum) is not int or maximum < 0):
        raise ProtocolError("budget.max_hardware_actions must be non-negative or unlimited")
    if "finish_actions" in budget and (type(budget["finish_actions"]) is not int or budget["finish_actions"] != 1):
        raise ProtocolError("budget.finish_actions must preserve the public value 1")
    if "source" in experiment:
        _string(experiment["source"], "experiment.source")
    if "method_options" in experiment:
        _mapping(experiment["method_options"], "experiment.method_options")
    result = dict(experiment)
    result["optimizer"] = optimizer
    result["budget"] = budget
    return result


def validate_top_level(value: Any, operation: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("request must be an object")
    request = dict(value)
    if request.get("schema_version") != 1:
        raise ProtocolError("schema_version must be 1")
    _reject_sensitive_deep(request, "request")
    if operation == "preflight":
        if set(request) != {"schema_version", "experiment"}:
            raise ProtocolError("preflight request has unsupported fields")
    elif operation == "prepare":
        if set(request) != {"schema_version", "experiment", "preflight", "context"}:
            raise ProtocolError("prepare request has unsupported fields")
        if not isinstance(request["preflight"], Mapping) or request["preflight"].get("schema_version") != 1:
            raise ProtocolError("preflight response is invalid")
    elif operation == "collect":
        allowed = {"schema_version", "experiment", "preflight", "context", "started", "close_reason"}
        if set(request) - allowed or not {"schema_version", "experiment", "preflight", "context"} <= set(request):
            raise ProtocolError("collect request has unsupported fields")
        if not isinstance(request["preflight"], Mapping) or request["preflight"].get("schema_version") != 1:
            raise ProtocolError("preflight response is invalid")
    elif operation == "run":
        allowed = {"schema_version", "experiment", "preflight", "context", "started", "close_reason"}
        if set(request) - allowed or not {"schema_version", "experiment", "preflight", "context"} <= set(request):
            raise ProtocolError("run request has unsupported fields")
        if not isinstance(request["preflight"], Mapping) or request["preflight"].get("schema_version") != 1:
            raise ProtocolError("preflight response is invalid")
    else:
        raise ProtocolError("unknown operation")
    request["experiment"] = validate_experiment(request.get("experiment"))
    return request


def response(**fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, **fields}


def error_response(error: BaseException) -> dict[str, Any]:
    code = getattr(error, "error_code", "infrastructure_error")
    if code not in {"invalid_request", "infrastructure_error"}:
        code = "infrastructure_error"
    return response(error_code=code, message=str(error)[:240])
