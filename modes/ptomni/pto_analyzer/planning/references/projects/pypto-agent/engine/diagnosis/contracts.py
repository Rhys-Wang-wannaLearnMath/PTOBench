"""数据契约：DiagnosisRecord、归一化泳道、ActionPlan、MeasuredResult。"""

DIAGNOSIS_SCHEMA = "pypto-diagnosis-1.0"
SWIMLANE_SCHEMA = "pypto-swimlane-1.0"
ACTIONPLAN_SCHEMA = "pypto-actionplan-1.0"

SYMPTOM_BUBBLE_HIGH = "bubble_high"
SYMPTOM_CORE_UTIL_LOW = "core_util_low"
SYMPTOM_LOAD_IMBALANCE = "load_imbalance"
SYMPTOM_SINGLE_TASK_LONG = "single_task_long"
SYMPTOM_WALLTIME_GT_E2E = "walltime_gt_e2e"
SYMPTOM_CV_DDR_BYPASS = "cv_ddr_bypass"
SYMPTOM_STRUCTURAL_CEILING = "structural_ceiling"

SYMTOM_LABELS = {
    SYMPTOM_BUBBLE_HIGH: "气泡率 > 10%",
    SYMPTOM_CORE_UTIL_LOW: "平均核心利用率 < 50%",
    SYMPTOM_LOAD_IMBALANCE: "负载不均衡（AicoreTime 差异 > 20%）",
    SYMPTOM_SINGLE_TASK_LONG: "单 task 耗时过长",
    SYMPTOM_WALLTIME_GT_E2E: "wall time >> AICore E2E Time",
    SYMPTOM_CV_DDR_BYPASS: "CV 间数据经 DDR 中转",
    SYMPTOM_STRUCTURAL_CEILING: "结构天花板候选（等前驱>70% 且 任务/核<3——旋钮难兑现，评估 Stage3/止损）",
}

CV_PATH_OPCODES = {"L0C_COPY_UB", "UB_COPY_L1", "UB_COPY_ND2NZ"}
CV_SYNC_OPCODES = {"CV_SYNC", "SYNC"}
DDR_RELAY_OUT = "COPY_OUT"
DDR_RELAY_IN = "COPY_IN"
SPILL_MARKER = "WorkspaceGm"
SPILL_EMPIRICAL_LIMIT = 20


def empty_diagnosis(op_id, soc, artifacts_dir):
    return {
        "schema": DIAGNOSIS_SCHEMA,
        "op_id": op_id,
        "soc": soc,
        "artifacts_dir": str(artifacts_dir),
        "l1_symptoms": [],
        "l1_metrics": {},
        "env_meta": {},
        "l2_localization": [],
        "l2_merge_hints": [],
        "l2_core_status": [],
        "l2_occupancy": {},
        "l2_structure": {},
        "l3_attribution": [],
        "l4_findings": [],
        "census": {},
        "symptoms": [],
        "hypotheses": [],
        "layers_available": {},
    }


def empty_normalized_swimlane(op_id, soc):
    return {
        "schema": SWIMLANE_SCHEMA,
        "op_id": op_id,
        "soc": soc,
        "cores": [],
        "merge_hints": [],
        "spills": [],
    }


def make_action(action_id, catalog_record, action_type, preconditions, gate_status, evidence_refs, coupled_group=None, offline=True):
    prereq_ready = all(p.get("state") == "satisfied" for p in preconditions)
    platform_ready = gate_status.get("platform_ok", False)
    if not platform_ready:
        status = "platform_not_applicable"
    elif not prereq_ready:
        status = "blocked_prerequisite"
    else:
        status = "pending_npu_validation" if offline else "ready"
    return {
        "id": action_id,
        "title": catalog_record.get("title", action_id),
        "stage": catalog_record.get("stage"),
        "action_type": action_type,
        "priority": catalog_record.get("priority", 1),
        "preconditions": preconditions,
        "gate_status": gate_status,
        "coupled_group": coupled_group,
        "evidence_refs": evidence_refs,
        "knob_target": catalog_record.get("knob_target"),
        "config_example": catalog_record.get("config_example"),
        "guide_ref": catalog_record.get("guide_ref"),
        "disciplines": catalog_record.get("disciplines", []),
        "status": status,
    }
