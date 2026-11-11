"""诊断状态机：编排 L1→L2→L3→L4，执行证据门，生成假设。"""

from pathlib import Path

from . import l1_symptom, l2_localize, l3_attribute, l4_compiler
from . import contracts


def _evidence_gate(symptom, diagnosis):
    needed = ["L1"]
    if symptom["name"] in (contracts.SYMPTOM_LOAD_IMBALANCE, contracts.SYMPTOM_SINGLE_TASK_LONG):
        needed.append("L2")
    if symptom["name"] == contracts.SYMPTOM_CV_DDR_BYPASS:
        needed.append("L4")
    return needed


def _build_hypotheses(diagnosis):
    hypotheses = []
    seen = set()
    for symptom in diagnosis["symptoms"]:
        if symptom["name"] in seen:
            continue
        seen.add(symptom["name"])
        hypotheses.append({
            "id": f"H{len(hypotheses) + 1}",
            "symptom": symptom["name"],
            "evidence": symptom["evidence"],
            "layers_required": _evidence_gate(symptom, diagnosis),
            "confidence": "measured" if symptom["detector"] in ("L1", "L2", "L4") else "speculative",
            "status": "pending",
        })
    diagnosis["hypotheses"] = hypotheses
    return diagnosis


def run(artifacts_dir, op_id, soc, source_dir=None, run_l3=True):
    artifacts = Path(artifacts_dir)
    diagnosis = contracts.empty_diagnosis(op_id, soc, artifacts)
    normalized = None

    diagnosis = l1_symptom.run(artifacts, diagnosis)
    diagnosis, normalized = l2_localize.run(artifacts, diagnosis)
    if run_l3 and normalized is not None:
        diagnosis = l3_attribute.run(artifacts, diagnosis, source_dir)
    diagnosis = l4_compiler.run(artifacts, diagnosis)

    # Stage 2 核内证据：指令 Census（computation_graph 主 dump 存在时）
    from static_analysis import census as census_mod
    try:
        census = census_mod.analyze_census(artifacts)
        if census:
            diagnosis["layers_available"]["census"] = True
            diagnosis["census"] = census
            diagnosis["l4_findings"].extend(census_mod.census_findings(census))
    except (ValueError, OSError):
        pass

    diagnosis = _build_hypotheses(diagnosis)
    diagnosis["artifacts_present"] = {
        "bubble_analysis.log": (artifacts / "bubble_analysis.log").exists(),
        "merged_swimlane.json": (artifacts / "merged_swimlane.json").exists(),
        "program.json": (artifacts / "program.json").exists(),
        "compile_stdout.log": (artifacts / "compile_stdout.log").exists(),
    }
    return diagnosis, normalized
