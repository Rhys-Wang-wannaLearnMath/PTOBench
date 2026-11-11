"""Public RunContext and path policy.  No evaluator internals are imported."""

from __future__ import annotations

from dataclasses import dataclass
import os
import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import ProtocolError


_FORBIDDEN_COMPONENTS = {"sealed", "sealed_assets", "evaluator", "control", "golden", "launcher"}


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _absolute_directory(value: Any, name: str, *, must_exist: bool = True,
                        check_forbidden: bool = True) -> Path:
    if type(value) is not str or not value:
        raise ProtocolError(f"context.{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ProtocolError(f"context.{name} must be an absolute normalized path")
    if _has_symlink_component(path):
        raise ProtocolError(f"context.{name} must not be a symlink")
    if must_exist and (not path.exists() or not path.is_dir()):
        raise ProtocolError(f"context.{name} is unavailable")
    if check_forbidden and any(part in _FORBIDDEN_COMPONENTS for part in path.parts):
        raise ProtocolError(f"context.{name} is outside the public run area")
    return path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class RunContext:
    session_root: Path
    workspace: Path
    session_id: str
    case_id: str
    case_module: str
    case_entry: str
    budget: Mapping[str, Any]
    profiling_modes: tuple[str, ...]

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> "RunContext":
        value = request.get("context")
        if not isinstance(value, Mapping):
            raise ProtocolError("context must be an object")
        allowed = {"session_root", "session_id", "workspace", "budget", "profiling_modes", "case"}
        if set(value) != allowed:
            raise ProtocolError("context has unsupported or missing fields")
        # A Session may itself live below a repository directory named
        # ``evaluator`` or ``launcher``.  The restriction applies to public
        # paths inside the Session, never to an unrelated parent component.
        root = _absolute_directory(value["session_root"], "session_root", check_forbidden=False)
        workspace = _absolute_directory(value["workspace"], "workspace", check_forbidden=False)
        if not _inside(workspace, root):
            raise ProtocolError("workspace is outside session_root")
        relative_parts = workspace.relative_to(root).parts
        if any(part in _FORBIDDEN_COMPONENTS for part in relative_parts):
            raise ProtocolError("workspace is outside the public run area")
        if type(value["session_id"]) is not str or not value["session_id"]:
            raise ProtocolError("context.session_id is invalid")
        budget = value["budget"]
        if not isinstance(budget, Mapping) or any(not isinstance(k, str) for k in budget):
            raise ProtocolError("context.budget is invalid")
        maximum = budget.get("max_hardware_actions")
        if maximum != "unlimited" and (type(maximum) is not int or maximum < 0):
            raise ProtocolError("context.budget has an invalid public limit")
        modes = value["profiling_modes"]
        if not isinstance(modes, (list, tuple)) or any(type(m) is not str or not m for m in modes):
            raise ProtocolError("context.profiling_modes is invalid")
        case = value["case"]
        if not isinstance(case, Mapping) or set(case) != {"case_id", "module", "entry"}:
            raise ProtocolError("context.case is invalid")
        for key in ("case_id", "module", "entry"):
            if type(case[key]) is not str or not case[key]:
                raise ProtocolError(f"context.case.{key} is invalid")
        experiment = request.get("experiment")
        if isinstance(experiment, Mapping) and experiment.get("budget") != dict(budget):
            raise ProtocolError("context budget does not match experiment")
        if isinstance(experiment, Mapping) and experiment.get("case") != case["case_id"]:
            raise ProtocolError("context Case does not match experiment")
        return cls(root, workspace, value["session_id"], case["case_id"], case["module"],
                   case["entry"], dict(budget), tuple(modes))

    @property
    def method_root(self) -> Path:
        return self.session_root / "method"

    @property
    def work_root(self) -> Path:
        return self.method_root / "work"

    def validate_workspace(self, workspace: Path | None = None) -> Path:
        candidate = self.workspace if workspace is None else workspace
        if not candidate.is_absolute() or _has_symlink_component(candidate) or not candidate.is_dir():
            raise ProtocolError("workspace is invalid")
        candidate = candidate.absolute()
        if not _inside(candidate, self.session_root):
            raise ProtocolError("workspace is outside session_root")
        # Public context exposes registered workspace roots.  A method may
        # select a sibling returned by ``fork`` but may not turn an arbitrary
        # subdirectory into a new workspace.
        if candidate.parent != self.workspace.parent:
            raise ProtocolError("workspace is not a registered workspace root")
        if any(part in _FORBIDDEN_COMPONENTS for part in candidate.relative_to(self.session_root).parts):
            raise ProtocolError("workspace is outside the public run area")
        if candidate != self.workspace:
            binding_path = candidate / ".benchmark_context/workspace.json"
            if _has_symlink_component(binding_path):
                raise ProtocolError("fork workspace binding must not be a symlink")
            try:
                binding = json.loads(binding_path.read_text())
            except (OSError, ValueError) as exc:
                raise ProtocolError("fork workspace has no public registration binding") from exc
            if binding.get("case_id") != self.case_id or binding.get("workspace") != str(candidate):
                raise ProtocolError("fork workspace public registration does not match")
        return candidate

    def public_dict(self) -> dict[str, Any]:
        return {
            "session_root": str(self.session_root), "workspace": str(self.workspace),
            "session_id": self.session_id, "case": {"case_id": self.case_id,
            "module": self.case_module, "entry": self.case_entry},
            "budget": dict(self.budget), "profiling_modes": list(self.profiling_modes),
        }

    def public_context_root(self, workspace: Path) -> Path:
        """Accept the broker's exact read-only public-contract mount link.

        Ordinary host paths still forbid links. Kernel isolation exports this
        one contract link per registered workspace; arbitrary links are refused.
        """
        workspace = self.validate_workspace(workspace)
        root = workspace / '.benchmark_context'
        if root.is_symlink():
            target = Path('/ptobench/public') / workspace.name
            if root.readlink() != target or _has_symlink_component(target) or not target.is_dir():
                raise ValueError('public_path_invalid')
            return target
        if _has_symlink_component(root):
            raise ValueError('public_path_invalid')
        return root


def in_current_process_group() -> bool:
    """Validate an optional launcher-provided group marker without guessing one."""
    marker = os.environ.get("PTOBENCH_METHOD_PGID")
    if marker is None:
        return True
    try:
        return os.getpgid(os.getpid()) == int(marker)
    except (ValueError, OSError):
        return False
