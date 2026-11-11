"""L2 子图定位：解析 merged_swimlane.json（perfetto 风格 trace），归一化泳道并提取合并线索。

输入契约（对齐 tune-swimlane/scripts/analyze_swimlane.py）：
- 顶层 {"traceEvents": [...]} 或事件列表
- 事件含 ts/dur/tid/args；args 内 hashOrder-hint 提供
  l1ReuseInfo / cubeMergeInfo / vecMergeInfo（各含 hashOrder 与 subGraphCount）
- 核类型（AIC/AIV）来自 tid 元数据事件或 args
"""

import json
from functools import lru_cache
from pathlib import Path

from . import contracts

HINT_KEYS = {
    "l1ReuseInfo": "l1Reuse",
    "cubeMergeInfo": "cubeMerge",
    "vecMergeInfo": "vecMerge",
}
HARDWARE_PROFILE = Path(__file__).resolve().parents[1] / "hardware" / "profile.json"


def _iter_events(raw):
    if isinstance(raw, dict):
        for key in ("traceEvents", "events", "trace_events"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return []
    if isinstance(raw, list):
        return raw
    return []


def _extract_hint(args):
    hint = args.get("hashOrder-hint") or args.get("hashOrder_hint") or {}
    if isinstance(hint, str):
        try:
            hint = json.loads(hint)
        except (ValueError, TypeError):
            hint = {}
    return hint if isinstance(hint, dict) else {}


def _core_type(event, tid_names):
    args = event.get("args") or {}
    for key in ("coreType", "core_type", "core"):
        if args.get(key):
            value = str(args[key]).upper()
            if "AIC" in value:
                return "AIC"
            if "AIV" in value:
                return "AIV"
    name = str(event.get("name", ""))
    if "AIC" in name:
        return "AIC"
    if "AIV" in name:
        return "AIV"
    tid_name = tid_names.get(event.get("tid"))
    if tid_name:
        if "AIC" in tid_name:
            return "AIC"
        if "AIV" in tid_name:
            return "AIV"
    return None


def _leaf_hash(event):
    args = event.get("args") or {}
    for key in ("leafHash", "leaf_hash", "hash"):
        if args.get(key):
            return str(args[key])
    return str(event.get("name", "unknown"))


@lru_cache(maxsize=32)
def _parse_swimlane_cached(path, mtime_ns, size):
    del mtime_ns, size
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    events = _iter_events(raw)
    tid_names = {}
    for ev in events:
        name = ev.get("name", "")
        if name in ("thread_name", "process_name"):
            tid = (ev.get("args") or {}).get("tid", ev.get("tid"))
            label = (ev.get("args") or {}).get("name", "")
            if tid is not None:
                tid_names[tid] = str(label)

    tasks_by_core = {}
    hints = {}
    spills = []
    for ev in events:
        args = ev.get("args") or {}
        if contracts.SPILL_MARKER in str(ev.get("name", "")) or contracts.SPILL_MARKER in str(args.get("name", "")):
            spills.append({"source": "swimlane", "name": ev.get("name")})
        dur = ev.get("dur")
        if not isinstance(dur, (int, float)) or dur <= 0:
            continue
        ctype = _core_type(ev, tid_names)
        if ctype is None:
            continue
        core_id = str((ev.get("args") or {}).get("coreId") or ev.get("tid"))
        if core_id not in tid_names and ev.get("tid") in tid_names:
            core_id = tid_names[ev.get("tid")]
        entry = {
            "start_us": float(ev.get("ts", 0)),
            "dur_us": float(dur),
            "leaf_hash": _leaf_hash(ev),
            "psg_id": args.get("psgId"),
            "root_name": args.get("rootName") or args.get("root_name") or ev.get("cat"),
        }
        tasks_by_core.setdefault((ctype, str(core_id)), []).append(entry)
        hint = _extract_hint(args)
        for src_key, hint_type in HINT_KEYS.items():
            info = hint.get(src_key)
            if isinstance(info, dict) and info.get("hashOrder") is not None:
                key = (hint_type, str(info.get("hashOrder")), entry["leaf_hash"])
                agg = hints.setdefault(key, {
                    "hint_type": hint_type,
                    "hash_order": str(info.get("hashOrder")),
                    "leaf_hash": entry["leaf_hash"],
                    "core_type": ctype,
                    "sub_graph_count": int(info.get("subGraphCount", 0) or 0),
                    "count": 0, "total_us": 0.0,
                })
                agg["count"] += 1
                agg["total_us"] += entry["dur_us"]
    return tasks_by_core, list(hints.values()), spills


def parse_swimlane(path):
    path = Path(path)
    stat = path.stat()
    return _parse_swimlane_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def normalize(op_id, soc, tasks_by_core, merge_hints, spills):
    cores = []
    for (ctype, core_id), tasks in tasks_by_core.items():
        tasks = sorted(tasks, key=lambda t: t["start_us"])
        segments = []
        cursor = None
        for t in tasks:
            if cursor is not None and t["start_us"] > cursor + 1e-9:
                segments.append({"kind": "gap", "start_us": cursor, "dur_us": t["start_us"] - cursor})
            segments.append({"kind": "task", **t})
            cursor = max(cursor or 0, t["start_us"] + t["dur_us"])
        cores.append({
            "core_id": core_id,
            "core_type": ctype,
            "segments": segments,
            "task_count": len(tasks),
        })
    cores.sort(key=lambda c: (c["core_type"], c["core_id"]))
    normalized_hints = []
    for raw_hint in merge_hints:
        hint = dict(raw_hint)
        if hint["count"]:
            hint["avg_us"] = round(hint["total_us"] / hint["count"], 3)
        hint["total_us"] = round(hint["total_us"], 3)
        normalized_hints.append(hint)
    result = contracts.empty_normalized_swimlane(op_id, soc)
    result["cores"] = cores
    result["merge_hints"] = normalized_hints
    result["spills"] = spills
    return result


def leaf_stats(normalized):
    stats = {}
    for core in normalized["cores"]:
        if core["core_type"] != "AIC":
            continue
        for seg in core["segments"]:
            if seg["kind"] != "task":
                continue
            agg = stats.setdefault(seg["leaf_hash"], {
                "leaf_hash": seg["leaf_hash"],
                "core_type": core["core_type"],
                "psg_id": seg.get("psg_id"),
                "root_name": seg.get("root_name"),
                "count": 0, "total_us": 0.0, "max_us": 0.0,
            })
            agg["count"] += 1
            agg["total_us"] += seg["dur_us"]
            agg["max_us"] = max(agg["max_us"], seg["dur_us"])
    for agg in stats.values():
        agg["avg_us"] = round(agg["total_us"] / agg["count"], 3) if agg["count"] else 0.0
        agg["total_us"] = round(agg["total_us"], 3)
    return sorted(stats.values(), key=lambda a: -a["total_us"])


@lru_cache(maxsize=1)
def _validated_core_counts():
    try:
        profile = json.loads(HARDWARE_PROFILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    result = {}
    aiv = profile.get("constants", {}).get("aiv_cores", {})
    if aiv.get("status") == "validated" and isinstance(aiv.get("value"), int):
        result["AIV"] = aiv["value"]
    return result


def _core_occupancy(normalized, expected_cores=None):
    """Infer occupancy per core type, preserving unknown hardware counts."""
    if expected_cores is None:
        soc = normalized.get("soc")
        expected_cores = _validated_core_counts() if soc in ("DAV_3510", "DAV_C310", "A5") else {}
    result = {}
    for ctype in ("AIC", "AIV"):
        observed = [c for c in normalized["cores"] if c["core_type"] == ctype]
        expected = expected_cores.get(ctype)
        if not observed:
            state = "UNKNOWN"
        elif expected is None:
            state = "UNKNOWN"
        elif len(observed) >= expected and all(c["task_count"] > 0 for c in observed):
            state = "FULL"
        else:
            state = "NOT_FULL"
        result[ctype] = {
            "state": state,
            "observed_cores": len(observed),
            "expected_cores": expected,
        }
    return result


def detect_l2_symptoms(normalized, diagnosis):
    stats = leaf_stats(normalized)
    single_task = [s for s in stats if s["count"] >= 1 and s["max_us"] > 0]
    if single_task:
        worst = single_task[0]
        total_busy = sum(s["total_us"] for s in stats)
        if total_busy and worst["total_us"] / total_busy > 0.5:
            diagnosis["symptoms"].append({
                "name": contracts.SYMPTOM_SINGLE_TASK_LONG,
                "detector": "L2",
                "evidence": f"merged_swimlane.json#leaf:{worst['leaf_hash']}",
            })
    occupancy = _core_occupancy(normalized)
    diagnosis["l2_occupancy"] = occupancy
    for core in normalized["cores"]:
        info = occupancy[core["core_type"]]
        diagnosis["l2_core_status"].append({
            "core_id": core["core_id"], "core_type": core["core_type"],
            "task_count": core["task_count"], "status": info["state"],
            "observed_cores": info["observed_cores"],
            "expected_cores": info["expected_cores"],
        })
    return diagnosis


def structure_stats(normalized):
    """泳道结构统计（pypto 0.2.1 无 hashOrder-hint 时的合并状态代理指标）。

    - task_count_per_core：每核任务数（中位/最小/最大）——合并状态的直接体现
      （合并粒度 = 子图总数/核数 恰好整除时每核 1 个大任务）
    - cold_start_phase：任一核的首个任务时长 > 3× 该核任务时长中位 → 视为冷启相位
      （跨 run 对比前应剥离每核头部冷启任务，否则时长分布不可比）
    """
    per_type = {}
    cold_heads = 0
    cold_cores = 0
    for core in normalized["cores"]:
        ctype = core["core_type"]
        tasks = [s for s in core["segments"] if s["kind"] == "task"]
        if not tasks:
            continue
        agg = per_type.setdefault(ctype, {"core_count": 0, "task_counts": []})
        agg["core_count"] += 1
        agg["task_counts"].append(len(tasks))
        if len(tasks) >= 3:
            durs = sorted(t["dur_us"] for t in tasks)
            med = durs[len(durs) // 2]
            head = tasks[0]["dur_us"]
            if med > 0 and head > 3 * med:
                cold_cores += 1
                cold_heads += 1
    stats = {}
    for ctype, agg in per_type.items():
        counts = sorted(agg["task_counts"])
        stats[ctype] = {
            "core_count": agg["core_count"],
            "task_count_per_core_med": counts[len(counts) // 2],
            "task_count_per_core_min": counts[0],
            "task_count_per_core_max": counts[-1],
        }
    stats["cold_start_phase"] = cold_cores > 0
    stats["cold_start_head_cores"] = cold_cores
    return stats


def run(artifacts_dir, diagnosis):
    path = Path(artifacts_dir) / "merged_swimlane.json"
    if not path.exists():
        diagnosis["layers_available"]["L2"] = False
        return diagnosis, None
    try:
        tasks_by_core, merge_hints, spills = parse_swimlane(path)
    except (ValueError, OSError):
        diagnosis["layers_available"]["L2"] = "parse_error"
        return diagnosis, None
    if not tasks_by_core:
        diagnosis["layers_available"]["L2"] = "no_events"
        return diagnosis, None
    normalized = normalize(diagnosis["op_id"], diagnosis["soc"], tasks_by_core, merge_hints, spills)
    diagnosis["layers_available"]["L2"] = True
    diagnosis["l2_localization"] = leaf_stats(normalized)[:10]
    diagnosis["l2_merge_hints"] = normalized["merge_hints"]
    diagnosis["l2_structure"] = structure_stats(normalized)
    diagnosis = detect_l2_symptoms(normalized, diagnosis)
    return diagnosis, normalized
