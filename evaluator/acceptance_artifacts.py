"""Validate native profiling evidence, using the schemas certified in S6."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def inspect_artifacts(root: Path, mode: str) -> dict:
    report = {"passed": False, "mode": mode, "root": str(root), "evidence": [], "errors": []}
    if not root.is_dir() or root.is_symlink():
        report["errors"].append("artifact_directory_missing_or_link")
        return report

    def files(pattern):
        result = []
        for path in sorted(root.rglob(pattern)):
            if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
                report["errors"].append(f"artifact_symlink:{path}")
            elif path.is_file() and path.stat().st_size:
                result.append(path)
        return result

    def require(pattern, validate):
        paths = files(pattern)
        if not paths:
            report["errors"].append(f"artifact_missing:{pattern}")
        for path in paths:
            try:
                schema = validate(path)
                report["evidence"].append({"path": str(path), "bytes": path.stat().st_size, "schema": schema})
            except (OSError, ValueError, TypeError, KeyError, AssertionError) as error:
                report["errors"].append(f"artifact_invalid:{path}:{error}")

    def mapping(path, required):
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or any(not isinstance(value.get(k), t) for k, t in required.items()):
            raise ValueError("required schema fields missing")
        return value, {k: type(value[k]).__name__ for k in required}

    if mode == "runtime":
        def swimlane(path):
            value, schema = mapping(path, {"aicore_tasks": list, "aicpu_tasks": list,
                "aicpu_orchestrator_phases": list, "aicpu_scheduler_phases": list,
                "chip_swimlane_level": int, "metadata": dict})
            if not value["aicore_tasks"] and not value["aicpu_tasks"]:
                raise ValueError("empty scheduling evidence")
            return schema

        def dependencies(path):
            value, schema = mapping(path, {"edges": list, "tasks": list, "tensors": list})
            if not value["tasks"]:
                raise ValueError("empty dependency tasks")
            return schema

        def trace(path):
            value, schema = mapping(path, {"traceEvents": list})
            if not value["traceEvents"]:
                raise ValueError("empty trace")
            return schema

        def scopes(path):
            header = {"total", "fatal", "dropped", "heap_max", "dep_pool_max",
                      "task_window_max", "tensormap_max"}
            fields = {"site", "phase", "depth", "ring", "heap_start", "heap_end",
                      "dep_pool_start", "dep_pool_end", "task_window_start", "task_window_end", "tensormap"}
            count, metadata = 0, None
            with path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("scope record is not an object")
                    if metadata is None:
                        if not header <= row.keys() or type(row["fatal"]) is not bool:
                            raise ValueError("scope header incomplete")
                        if any(type(row[k]) is not int or row[k] < 0 for k in ("total", "dropped", "tensormap_max")):
                            raise ValueError("invalid scope header counters")
                        for k in ("heap_max", "dep_pool_max", "task_window_max"):
                            if not isinstance(row[k], list) or not row[k] or any(type(v) is not int or v < 0 for v in row[k]):
                                raise ValueError("invalid scope capacity")
                        metadata = row
                        continue
                    if not fields <= row.keys() or not isinstance(row["site"], str) or row["phase"] not in {"begin", "end"}:
                        raise ValueError("scope event schema incomplete")
                    if any(type(row[k]) is not int or row[k] < 0 for k in fields - {"site", "phase"}):
                        raise ValueError("invalid scope event counters")
                    count += 1
            if not count or metadata is None or metadata["fatal"] or metadata["total"] != count + metadata["dropped"]:
                raise ValueError("empty, fatal or inconsistent scope records")
            return {"header_fields": sorted(header), "event_fields": sorted(fields), "records": count,
                    "dropped": metadata["dropped"]}

        require("chip_swimlane_records.json", swimlane)
        require("deps.json", dependencies)
        require("merged_swimlane_*.json", trace)
        require("scope_stats.jsonl", scopes)
    elif mode == "pmu":
        def counters(path):
            fields = {"thread_id", "core_id", "task_id", "func_id", "core_type", "pmu_total_cycles",
                "vec_busy_cycles", "cube_busy_cycles", "scalar_busy_cycles", "mte1_busy_cycles",
                "mte2_busy_cycles", "mte3_busy_cycles", "icache_miss", "icache_req", "event_type"}
            count = 0
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if not fields <= set(reader.fieldnames or []):
                    raise ValueError("PMU schema incomplete")
                for row in reader:
                    if None in row or any(row.get(k) is None for k in fields):
                        raise ValueError("malformed PMU row")
                    value = float(row["pmu_total_cycles"])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("invalid PMU cycle count")
                    count += 1
            if not count:
                raise ValueError("empty PMU counters")
            return {"fields": sorted(fields), "records": count}

        require("pmu.csv", counters)
    elif mode == "compiler":
        def compilation(path):
            value, schema = mapping(path, {"stages": list, "total_seconds": (int, float)})
            if not value["stages"] or not math.isfinite(value["total_seconds"]) or value["total_seconds"] < 0:
                raise ValueError("empty or invalid compilation profile")
            return schema

        require("pipeline_profile.json", compilation)
        dumps = files("*_after_*.py")
        frontends = files("00_frontend.py")
        if not dumps or not frontends:
            report["errors"].append("artifact_missing:compiler_pass_dumps")
        else:
            report["evidence"].append({"kind": "passes_dump", "paths": [str(p) for p in frontends + dumps]})
    elif mode != "clean":
        report["errors"].append("invalid_artifact_mode")
    report["passed"] = not report["errors"]
    return report
