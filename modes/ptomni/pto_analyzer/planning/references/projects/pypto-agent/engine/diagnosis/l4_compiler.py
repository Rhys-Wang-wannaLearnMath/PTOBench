"""L4 编译器行为分析：解析 program.json，追踪 CV 通路 / DDR 中转 / spill / 合图生效。

判定规则（源自 optimization_catalog.md S-14 验证方法）：
- CV 通路 opcode：L0C_COPY_UB / UB_COPY_L1 / UB_COPY_ND2NZ 为片上直连
- DDR 中转：COPY_OUT 后紧接 COPY_IN 且无 CV_SYNC 邻居
- 存在 CV 通路 opcode 只能证明部分数据走 CV 通路，须逐数据确认
- 正常 DDR 搬运（最终输出/原始输入/跨迭代依赖）不影响合图生效判断

pypto 0.2.1 真实 schema（首份样本 fixtures/rmsnorm/ 对齐）：
- 顶层为 {functions: [...], enable_cvfuse, version, entryhash, ...}
- 无扁平 op 列表；operations 为 CALL 级（kind=2/opcode="CALL"）
- functions[].rawname 携带 launch 层级（..._Unroll<N>_PATH0_hiddenfunc）
- functions[]._total_subgraph_count 携带同构子图组合并状态
→ 对该 schema 产出 launch 结构与合并状态发现（L2 无 hashOrder-hint 时的补充证据源）
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from . import contracts

_UNROLL_RE = re.compile(r"_Unroll(\d+)_")


def _extract_ops(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("ops", "opcodes", "instructions", "program", "tiles"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested = []
                for sub in value.values():
                    if isinstance(sub, list):
                        nested.extend(sub)
                if nested:
                    return nested
    return []


def _analyze_function_program(raw):
    """pypto 0.2.1 真实 schema：functions[] 结构 → launch 层级与合并状态。"""
    functions = raw.get("functions")
    if not isinstance(functions, list) or not functions:
        return None
    launches = []
    unroll_levels = set()
    for f in functions:
        if not isinstance(f, dict):
            continue
        name = str(f.get("rawname", ""))
        m = _UNROLL_RE.search(name)
        level = int(m.group(1)) if m else None
        if level:
            unroll_levels.add(level)
        launches.append({
            "rawname": name,
            "funcmagic": f.get("funcmagic"),
            "functype": f.get("functype"),
            "graphtype": f.get("graphtype"),
            "total_subgraph_count": f.get("_total_subgraph_count", 0),
            "sg_parallel_num": f.get("_sg_parallel_num"),
            "sg_partition_algorithm": f.get("_sg_partition_algorithm"),
        })
    subgraph_counts = [l["total_subgraph_count"] or 0 for l in launches]
    return {
        "schema_family": "pypto-0.2.1-functions",
        "function_count": len(functions),
        "unroll_levels": sorted(unroll_levels),
        "launch_functions": launches,
        "max_subgraph_count": max(subgraph_counts) if subgraph_counts else 0,
        "enable_cvfuse": raw.get("enable_cvfuse"),
        "program_version": raw.get("version"),
    }


def _opcode(op):
    for key in ("opcode", "op", "name", "opName"):
        if op.get(key):
            return str(op[key])
    return ""


def _semantic(op):
    args = op.get("args") or op.get("semantic") or {}
    for key in ("semantic_label", "semanticLabel", "label"):
        if args.get(key):
            return str(args[key])
    return ""


@lru_cache(maxsize=32)
def _analyze_program_cached(path, mtime_ns, size):
    del mtime_ns, size
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    func_result = _analyze_function_program(raw)
    ops = _extract_ops(raw)
    if not ops:
        if func_result:
            return func_result
        return {"parse_error": "program.json 中未找到 op 列表（尝试的键：ops/opcodes/instructions/program/tiles）"}

    cv_path_hits = []
    ddr_relays = []
    spill_count = 0
    for idx, op in enumerate(ops):
        code = _opcode(op)
        if code in contracts.CV_PATH_OPCODES:
            cv_path_hits.append({"op_index": idx, "opcode": code, "semantic": _semantic(op)})
        if contracts.SPILL_MARKER in code:
            spill_count += 1
        if code == contracts.DDR_RELAY_OUT:
            window = ops[idx + 1: idx + 4]
            window_codes = [_opcode(w) for w in window]
            has_cv_sync = any(c in contracts.CV_SYNC_OPCODES for c in window_codes)
            if contracts.DDR_RELAY_IN in window_codes and not has_cv_sync:
                ddr_relays.append({
                    "out_op_index": idx,
                    "in_op_index": idx + 1 + window_codes.index(contracts.DDR_RELAY_IN),
                    "semantic": _semantic(op),
                })
    result = {
        "op_count": len(ops),
        "cv_path_hits": cv_path_hits,
        "cv_path_count": len(cv_path_hits),
        "ddr_relays": ddr_relays,
        "ddr_relay_count": len(ddr_relays),
        "spill_count": spill_count,
        "spill_over_limit": spill_count > contracts.SPILL_EMPIRICAL_LIMIT,
    }
    if func_result:
        result.update({k: v for k, v in func_result.items() if k != "launch_functions"})
        result["launch_functions"] = func_result["launch_functions"]
    return result


def analyze_program(path):
    path = Path(path)
    stat = path.stat()
    return _analyze_program_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def run(artifacts_dir, diagnosis):
    path = Path(artifacts_dir) / "program.json"
    if not path.exists():
        diagnosis["layers_available"]["L4"] = False
        return diagnosis
    try:
        result = analyze_program(path)
    except (ValueError, OSError):
        diagnosis["layers_available"]["L4"] = "parse_error"
        return diagnosis
    if "parse_error" in result:
        diagnosis["layers_available"]["L4"] = "parse_error"
        diagnosis["l4_findings"].append({"kind": "parse_error", "detail": result["parse_error"]})
        return diagnosis

    diagnosis["layers_available"]["L4"] = True
    if result.get("schema_family") == "pypto-0.2.1-functions" or "unroll_levels" in result:
        levels = result.get("unroll_levels") or []
        max_sg = result.get("max_subgraph_count", 0)
        diagnosis["l4_findings"].append({
            "kind": "launch_structure",
            "detail": (
                f"{len(levels)} 级 unroll launch（{','.join(str(v) for v in levels)}），"
                f"program 含 {result.get('function_count')} 个 function"
                + (f"，最大同构子图组 subGraphCount={max_sg}" if max_sg else "，未见合并子图组")
                + f"，enable_cvfuse={result.get('enable_cvfuse')}"
            ),
            "evidence": (
                f"program.json#functions[{result.get('function_count')}],"
                f"unroll_levels={levels},max_subgraph_count={max_sg}"
            ),
        })
        if len(levels) >= 3:
            diagnosis["l4_findings"].append({
                "kind": "multi_level_launch",
                "detail": (
                    f"多层 unroll（{len(levels)} 级）产生多个非同构子图组——"
                    "nbuffer 合并无法跨组进行；若目标是压缩派发单元数，"
                    "考虑单层 unroll + 合并粒度 = 子图总数/核数"
                ),
                "evidence": f"program.json#unroll_levels={levels}",
            })
    if result.get("ddr_relay_count", 0) > 0:
        for relay in result["ddr_relays"]:
            diagnosis["l4_findings"].append({
                "kind": "cv_ddr_bypass",
                "detail": f"COPY_OUT@op[{relay['out_op_index']}] 紧接 COPY_IN@op[{relay['in_op_index']}] 无 CV_SYNC 邻居",
                "evidence": f"program.json:op[{relay['out_op_index']}]",
            })
        diagnosis["symptoms"].append({
            "name": contracts.SYMPTOM_CV_DDR_BYPASS,
            "detector": "L4",
            "evidence": f"program.json#ddr_relay_count={result['ddr_relay_count']}",
        })
    if result.get("cv_path_count", 0) > 0:
        diagnosis["l4_findings"].append({
            "kind": "cv_path_present",
            "detail": f"存在 {result['cv_path_count']} 个 CV 通路 opcode（注意：只能证明部分数据走 CV 通路，须逐数据确认）",
            "evidence": "program.json#cv_path_count",
        })
    if result.get("spill_count", 0) > 0:
        diagnosis["l4_findings"].append({
            "kind": "spill",
            "detail": f"spill（{contracts.SPILL_MARKER}）计数 {result['spill_count']}，经验阈值 {contracts.SPILL_EMPIRICAL_LIMIT}{'（超限，考虑调小 TileShape 或 nbuffer）' if result.get('spill_over_limit') else ''}",
            "evidence": "program.json#spill_count",
        })
    return diagnosis
