"""紧凑度分析：逐核忙闲比、间隙归因链（间隙 → 责任子图 → 建议优化点）。"""


def analyze(normalized, l1_metrics=None):
    results = []
    for core in normalized.get("cores", []):
        tasks = [s for s in core["segments"] if s["kind"] == "task"]
        gaps = [s for s in core["segments"] if s["kind"] == "gap"]
        busy = sum(t["dur_us"] for t in tasks)
        idle = sum(g["dur_us"] for g in gaps)
        total = busy + idle
        l1_core = None
        if l1_metrics:
            for c in l1_metrics.get("per_core", []):
                if c["core_name"] == core["core_id"]:
                    l1_core = c
                    break
        gap_kind = "unknown"
        if l1_core:
            schedule = l1_core.get("wait_schedule_time_us", 0)
            predecessor = l1_core.get("wait_predecessor_time_us", 0)
            if schedule + predecessor > 0:
                gap_kind = (
                    "wait_schedule" if schedule >= predecessor else "wait_predecessor"
                )
        results.append({
            "core_id": core["core_id"],
            "core_type": core["core_type"],
            "busy_us": round(busy, 3),
            "idle_us": round(idle, 3),
            "compactness_pct": round(busy / total * 100, 2) if total else 0.0,
            "gap_count": len(gaps),
            "dominant_gap_kind": gap_kind,
            "top_gaps": sorted(
                [{"start_us": g["start_us"], "dur_us": g["dur_us"]} for g in gaps],
                key=lambda g: -g["dur_us"],
            )[:3],
        })
    return results


def _gap_recommendation(gap_kind, core_type):
    if gap_kind == "wait_schedule":
        if core_type == "AIV":
            return "等待调度占比高：S-4 手动合图 / S-5 nbuffer 自动合图（连续短 AIV 任务）"
        return "等待调度占比高：S-3 负载均衡 / F-2/F-3 增大任务粒度 / S-20 max_workspace_kb"
    if gap_kind == "wait_predecessor":
        return "等待前驱占比高：瓶颈在上游子图——S-3 定位瓶颈子图，S-11/S-12 调其 TileShape"
    return "间隙类型未知（无 L1 数据或该核无等待记录）：检查子图依赖结构"


def format_report(compactness, top_gaps_per_core=3):
    lines = ["### 紧凑度分析（逐核忙闲比与间隙归因）", ""]
    lines.append("| 核心 | 类型 | 忙碌(us) | 空闲(us) | 紧凑度 | 间隙数 | 主导间隙类型 |")
    lines.append("|------|------|----------|----------|--------|--------|--------------|")
    for item in compactness:
        lines.append(
            "| {} | {} | {:.1f} | {:.1f} | {:.1f}% | {} | {} |".format(
                item["core_id"], item["core_type"], item["busy_us"], item["idle_us"],
                item["compactness_pct"], item["gap_count"], item["dominant_gap_kind"],
            )
        )
    lines.append("")
    all_gaps = []
    for item in compactness:
        for gap in item["top_gaps"][:top_gaps_per_core]:
            all_gaps.append((gap["dur_us"], item, gap))
    all_gaps.sort(key=lambda x: -x[0])
    lines.append("**Top 间隙归因链（按空闲时长降序）：**")
    lines.append("")
    for dur, item, gap in all_gaps[:5]:
        rec = _gap_recommendation(item["dominant_gap_kind"], item["core_type"])
        lines.append(
            "- `{}` @ {:.1f}us，时长 {:.1f}us（占该核总时 {:.0f}%）→ {} — *{}*".format(
                item["core_id"], gap["start_us"], dur,
                dur / (item["busy_us"] + item["idle_us"]) * 100 if (item["busy_us"] + item["idle_us"]) else 0,
                item["dominant_gap_kind"], rec,
            )
        )
    lines.append("")
    return "\n".join(lines)
