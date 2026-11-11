"""L1 症状聚合：解析 bubble_analysis.log，复用 analyze_perf.py 的指标口径。"""

import re
from pathlib import Path

from . import contracts

CORE_PATTERN = (
    r'\[(AIC_\d+|AIV_\d+)\] Execute task num:(\d+)\s+Core Total Work Time: ([\d.]+)\s+'
    r'Total Wait Time: ([\d.]+)\s+Wait Schedule Time: ([\d.]+)\s+'
    r'Wait Predecessor Time: ([\d.]+)'
)

WALLTIME_PATTERN = r'(?:wall|Wall)[^\d]*(?:time|Time)[^\d]*([\d.]+)\s*us'
E2E_HINT_PATTERN = r'(?:E2E|e2e|AICore Time|aicore_time)[^\d]*([\d.]+)\s*us'

# 环境指纹（可选尾行，跨 run 对比的环境门——驱动/CANN 代际变更可使同一配置
# 读数漂移 -45%，无指纹的跨 run 对比不可信）
ENV_PATTERNS = {
    "cann_version": r'^cann_version:\s*(.+)$',
    "pypto_version": r'^pypto_version:\s*(.+)$',
    "driver_note": r'^driver_note:\s*(.+)$',
}


# 理论 floor（可选尾行，人工填：最小流量÷带宽等——金标准④ Duration/floor 的分母）
FLOOR_PATTERN = r'^theoretical_floor:\s*([\d.]+)\s*us'


def parse_env_meta(log_text):
    meta = {}
    if not log_text:
        return meta
    for key, pattern in ENV_PATTERNS.items():
        m = re.search(pattern, log_text, re.MULTILINE)
        if m:
            meta[key] = m.group(1).strip()
    fm = re.search(FLOOR_PATTERN, log_text, re.MULTILINE)
    if fm:
        meta["theoretical_floor_us"] = float(fm.group(1))
    return meta


class CoreMetrics:
    def __init__(self, core_name, task_num, total_work_time, total_wait_time,
                 wait_schedule_time, wait_predecessor_time):
        self.core_name = core_name
        self.task_num = task_num
        self.total_work_time = total_work_time
        self.total_wait_time = total_wait_time
        self.wait_schedule_time = wait_schedule_time
        self.wait_predecessor_time = wait_predecessor_time

    @property
    def aicore_time(self):
        return self.total_work_time - self.total_wait_time

    @property
    def core_utilization(self):
        total = self.aicore_time + self.total_wait_time
        return (self.aicore_time / total * 100) if total else 0.0

    @property
    def bubble_rate(self):
        total = self.aicore_time + self.wait_schedule_time
        return (self.wait_schedule_time / total * 100) if total else 0.0

    def to_dict(self):
        return {
            "core_name": self.core_name,
            "task_num": self.task_num,
            "total_work_time_us": round(self.total_work_time, 3),
            "total_wait_time_us": round(self.total_wait_time, 3),
            "wait_schedule_time_us": round(self.wait_schedule_time, 3),
            "wait_predecessor_time_us": round(self.wait_predecessor_time, 3),
            "aicore_time_us": round(self.aicore_time, 3),
            "core_utilization_pct": round(self.core_utilization, 2),
            "bubble_rate_pct": round(self.bubble_rate, 2),
        }


def parse_bubble_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    cores = []
    for m in re.finditer(CORE_PATTERN, text):
        cores.append(CoreMetrics(
            m.group(1), int(m.group(2)), float(m.group(3)),
            float(m.group(4)), float(m.group(5)), float(m.group(6)),
        ))
    return cores, text


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def calculate_metrics(cores):
    aic = [c for c in cores if c.core_name.startswith("AIC")]
    aiv = [c for c in cores if c.core_name.startswith("AIV")]
    # Some vector-only operators have no AIC rows.  Use the longest observed
    # device lane as a conservative fallback; run() replaces this with the
    # profiler's explicit E2E value when the log provides one.
    device_cores = aic or cores
    aic_e2e = max((c.aicore_time + c.total_wait_time for c in device_cores), default=0.0)
    aic_times = [c.aicore_time for c in aic]
    imbalance_ratio = (max(aic_times) / min(aic_times)) if aic_times and min(aic_times) > 0 else 1.0

    # 等待构成分解（金标准②：区分"等调度"与"等前驱"）
    # 等调度=派发吞吐不足（旋钮可打）；等前驱=任务边界同步或跨阶段依赖（联合任务/核判读）
    def wait_composition(group):
        wait = sum(c.total_wait_time for c in group)
        sched = sum(c.wait_schedule_time for c in group)
        pred = sum(c.wait_predecessor_time for c in group)
        return {
            "total_wait_us": round(wait, 3),
            "dispatch_wait_pct": round(sched / wait * 100, 1) if wait > 0 else 0.0,
            "pred_wait_pct": round(pred / wait * 100, 1) if wait > 0 else 0.0,
        }

    return {
        "avg_core_utilization_pct": round(_avg([c.core_utilization for c in cores]), 2),
        "avg_bubble_rate_pct": round(_avg([c.bubble_rate for c in cores]), 2),
        "avg_aic_utilization_pct": round(_avg([c.core_utilization for c in aic]), 2),
        "avg_aiv_utilization_pct": round(_avg([c.core_utilization for c in aiv]), 2),
        "aic_e2e_time_us": round(aic_e2e, 3),
        "aic_load_imbalance_ratio": round(imbalance_ratio, 3),
        "max_core_work_time_us": round(max((c.total_work_time for c in cores), default=0.0), 3),
        "wait_composition": {
            "AIC": wait_composition(aic),
            "AIV": wait_composition(aiv),
        },
        "per_core": [c.to_dict() for c in cores],
    }


def parse_run_timing(log_text):
    """Return profiler-level wall/E2E timing without deriving it from lanes."""
    wall_match = re.search(WALLTIME_PATTERN, log_text or "")
    e2e_match = re.search(E2E_HINT_PATTERN, log_text or "")
    return {
        "wall_time_us": float(wall_match.group(1)) if wall_match else None,
        "aic_e2e_time_us": float(e2e_match.group(1)) if e2e_match else None,
    }


def detect_symptoms(metrics, log_text="", cores=None):
    symptoms = []

    def add(name, value, threshold, evidence):
        symptoms.append({
            "name": name,
            "label": contracts.SYMTOM_LABELS[name],
            "value": value,
            "threshold": threshold,
            "evidence": evidence,
            "detector": "L1",
        })

    if metrics["avg_bubble_rate_pct"] > 10:
        add(contracts.SYMPTOM_BUBBLE_HIGH, metrics["avg_bubble_rate_pct"], "> 10",
            "bubble_analysis.log#avg_bubble_rate")
    if metrics["avg_core_utilization_pct"] < 50:
        add(contracts.SYMPTOM_CORE_UTIL_LOW, metrics["avg_core_utilization_pct"], "< 50",
            "bubble_analysis.log#avg_core_utilization")
    if metrics["aic_load_imbalance_ratio"] > 1.2:
        add(contracts.SYMPTOM_LOAD_IMBALANCE, metrics["aic_load_imbalance_ratio"], "> 1.2",
            "bubble_analysis.log#aic_load_imbalance_ratio")

    # 金标准②联合判读：等前驱>70% 且 任务/核<3 → 结构天花板候选
    # （等前驱高但任务密度高 = 任务边界同步，合并可消——不触发本症状）
    wc = metrics.get("wait_composition", {})
    if cores:
        for ctype in ("AIC", "AIV"):
            comp = wc.get(ctype, {})
            grp = [c for c in cores if c.core_name.startswith(ctype)]
            task_density = (sum(c.task_num for c in grp) / len(grp)) if grp else 0.0
            if comp.get("pred_wait_pct", 0) > 70 and task_density < 3:
                add(contracts.SYMPTOM_STRUCTURAL_CEILING, comp["pred_wait_pct"], "> 70 且 任务/核<3",
                    "bubble_analysis.log#{}_pred_wait_pct={},task_density={:.1f}".format(
                        ctype, comp["pred_wait_pct"], task_density))
                break

    timing = parse_run_timing(log_text)
    wall = timing["wall_time_us"]
    e2e = timing["aic_e2e_time_us"]
    if wall and e2e and e2e > 0 and wall / e2e > 1.5:
        add(contracts.SYMPTOM_WALLTIME_GT_E2E, round(wall / e2e, 2), "> 1.5",
            "bubble_analysis.log#wall_vs_e2e")
    return symptoms


def run(artifacts_dir, diagnosis):
    log_path = Path(artifacts_dir) / "bubble_analysis.log"
    if not log_path.exists():
        diagnosis["layers_available"]["L1"] = False
        return diagnosis
    cores, log_text = parse_bubble_log(log_path)
    if not cores:
        diagnosis["layers_available"]["L1"] = False
        return diagnosis
    metrics = calculate_metrics(cores)
    timing = parse_run_timing(log_text)
    if timing["aic_e2e_time_us"] is not None:
        metrics["aic_e2e_time_us"] = timing["aic_e2e_time_us"]
        metrics["e2e_source"] = "bubble_analysis.log#AICore_E2E_Time"
    else:
        metrics["e2e_source"] = "derived_max_device_lane"
    if timing["wall_time_us"] is not None:
        metrics["wall_time_us"] = timing["wall_time_us"]
    diagnosis["layers_available"]["L1"] = True
    diagnosis["l1_metrics"] = metrics
    diagnosis["env_meta"] = parse_env_meta(log_text)
    diagnosis["l1_symptoms"] = detect_symptoms(metrics, log_text, cores=cores)
    diagnosis["symptoms"].extend(
        {"name": s["name"], "detector": "L1", "evidence": s["evidence"]}
        for s in diagnosis["l1_symptoms"]
    )
    return diagnosis
