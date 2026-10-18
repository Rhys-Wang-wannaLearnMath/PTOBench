"""Case-local packed prefill indexer ABI and comparator execution."""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from golden import ScalarSpec, TensorSpec, run
import common, workflow

_NAMES=("x","qr","qr_scale","wq_b","wq_b_scale","weights_proj","cos","sin","freqs_cos","freqs_sin","hadamard","inner_compress_state","inner_compress_state_block_table","inner_wkv","inner_wgate","inner_ape","inner_norm_w","idx_kv_cache","idx_kv_scale","idx_block_table","score","topk_idxs","position_ids","num_tokens","idx_slot_mapping","inner_state_slot_mapping")

def _config(case_dir, case):
    identity=case["case_id"].replace(".", "_")
    config=common.load_module(case_dir/case["config"], f"config_{identity}")
    if config.CASE_ID != case["case_id"]: raise ValueError("public config identity disagrees with Case")
    return config

def _num(value, name, *, upper=None):
    if type(value) not in (int,float) or not math.isfinite(float(value)) or value < 0 or (upper is not None and value > upper):
        raise ValueError(f"invalid comparator {name}")
    return float(value)

def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("duplicate comparator field")
        result[key]=value
    return result

def _policy(case_dir, case):
    value=json.loads((case_dir/case["comparator"]).read_text(encoding="utf-8"),object_pairs_hook=_unique_object)
    if not isinstance(value,dict) or set(value)!={"schema_version","outputs"} or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("invalid comparator top-level fields")
    outputs=value["outputs"]
    declared=set(case["correctness"]["outputs"])
    if not isinstance(outputs,dict) or set(outputs)!=declared: raise ValueError("comparator output coverage disagrees with contract")
    return outputs

def _ratio(rule):
    if set(rule)!={"method","rtol","atol","max_error_ratio"} or rule["method"]!="ratio_allclose": raise ValueError("invalid ratio comparator")
    policy_rtol=_num(rule["rtol"],"rtol"); policy_atol=_num(rule["atol"],"atol"); policy_ratio=_num(rule["max_error_ratio"],"max_error_ratio",upper=1)
    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape: return False,"output shape mismatch"
        a=actual.detach().cpu().float(); b=expected.detach().cpu().float()
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): return False,"nonfinite output"
        errors=int((a-b).abs().gt(policy_atol+policy_rtol*b.abs()).sum().item()); allowed=round(policy_ratio*a.numel())
        return errors<=allowed, "" if errors<=allowed else f"ratio_allclose: errors={errors}/{a.numel()} threshold={allowed}"
    compare.__name__="ratio_allclose"; return compare

def _mapped_active_rows(rule, fixed_shape):
    required={"method","name","mapping_name","mapping_shape","block_size","rtol","atol","max_error_ratio"}
    if set(rule)!=required or rule["method"]!="custom" or rule["name"]!="mapped_active_rows_ratio_allclose":
        raise ValueError("invalid mapped row comparator")
    if rule["mapping_name"]!="idx_slot_mapping" or rule["mapping_shape"]!=[128] or type(rule["mapping_shape"]) is not list or any(type(dim) is not int for dim in rule["mapping_shape"]) or type(rule["block_size"]) is not int or rule["block_size"]!=128:
        raise ValueError("mapped row contract disagrees with fixed input ABI")
    point=_ratio({"method":"ratio_allclose", **{key:rule[key] for key in ("rtol","atol","max_error_ratio")}})
    mapping_name=rule["mapping_name"]
    def compare(actual, expected, **kwargs):
        if tuple(actual.shape)!=fixed_shape or actual.shape!=expected.shape:
            return False,"physical pool shape mismatch"
        mapping=kwargs.get("inputs",{}).get(mapping_name)
        if mapping is None or mapping.dtype!=torch.int64 or list(mapping.shape)!=rule["mapping_shape"]:
            return False,"fixed mapping input missing or invalid"
        physical_rows=actual.shape[0]*actual.shape[1]
        mapping=mapping.detach().cpu().reshape(-1)
        active=torch.unique(mapping[mapping>=0],sorted=True)
        if active.numel() and int(active[-1])>=physical_rows:
            return False,"mapping exceeds physical pool"
        a=actual.detach().cpu().reshape(physical_rows,*actual.shape[2:])
        b=expected.detach().cpu().reshape(physical_rows,*expected.shape[2:])
        if active.numel():
            ok,detail=point(a.index_select(0,active),b.index_select(0,active),**kwargs)
            if not ok:return False,"active physical rows: "+detail
        inactive=torch.ones(physical_rows,dtype=torch.bool)
        inactive[active]=False
        if not torch.equal(a[inactive],b[inactive]):
            return False,"unmapped physical rows changed"
        return True,""
    compare.__name__=rule["name"]
    return compare

def _score(rule):
    required={"method","name","rtol","atol","max_error_ratio","tail_sentinel"}
    if set(rule)!=required or rule["method"]!="custom" or rule["name"]!="score_valid_rows_ratio_allclose": raise ValueError("invalid score comparator")
    policy_rtol=_num(rule["rtol"],"rtol"); policy_atol=_num(rule["atol"],"atol"); policy_ratio=_num(rule["max_error_ratio"],"max_error_ratio",upper=1)
    if type(rule["tail_sentinel"]) not in (int,float) or not math.isfinite(float(rule["tail_sentinel"])):
        raise ValueError("invalid score tail sentinel")
    sentinel=float(rule["tail_sentinel"])
    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape: return False,"score shape mismatch"
        inputs=kwargs.get("inputs",{}); token_value=inputs.get("num_tokens")
        if token_value is None: return False,"num_tokens input missing"
        n=int(token_value.item()) if hasattr(token_value,"item") else int(token_value)
        if n<0 or n>actual.shape[0]: return False,"num_tokens out of bounds"
        a=actual.detach().cpu().float(); b=expected.detach().cpu().float()
        if not bool(torch.isfinite(a[:n]).all() and torch.isfinite(b[:n]).all()): return False,"nonfinite active score"
        if a.shape[0]>n and bool((a[n:]!=sentinel).any()): return False,"non-sentinel inactive score tail"
        aa=a[:n]; bb=b[:n]
        errors=int((aa-bb).abs().gt(policy_atol+policy_rtol*bb.abs()).sum().item()); allowed=round(policy_ratio*aa.numel())
        return errors<=allowed, "" if errors<=allowed else f"score_valid_rows_ratio_allclose: errors={errors}/{aa.numel()} threshold={allowed}"
    compare.__name__=rule["name"]; return compare

def _topk(rule, config):
    required={"method","name","paired_output","valid_prefix","descending","tie_policy","padding"}
    if set(rule)!=required or rule["method"]!="custom" or rule["name"]!="topk_pair_compare":
        raise ValueError("invalid topk comparator")
    if (rule["paired_output"]!="score" or type(rule["valid_prefix"]) is not int
        or rule["valid_prefix"]!=config.INDEXER_TOPK_CAP or rule["descending"] is not True
        or rule["tie_policy"]!="actual_score_monotonic" or type(rule["padding"]) is not int
        or rule["padding"]!=-1):
        raise ValueError("invalid topk policy")
    paired_output=rule["paired_output"]; prefix_cap=rule["valid_prefix"]; padding=rule["padding"]
    def compare(actual, expected, *, actual_outputs, **kwargs):
        inputs=kwargs.get("inputs", {})
        positions=inputs.get("position_ids"); token_value=inputs.get("num_tokens")
        if positions is None or token_value is None: return False,"topk metadata missing"
        n=int(token_value.item()) if hasattr(token_value,"item") else int(token_value)
        if (actual.shape!=expected.shape or tuple(actual.shape)!=(config.T,config.INDEXER_SCORE_CAP)
            or list(positions.shape)!=[config.T] or n<0 or n>config.T):
            return False,"topk shape or metadata mismatch"
        score=actual_outputs.get(paired_output)
        if score is None or score.shape!=actual.shape: return False,"paired score missing"
        a=actual.detach().cpu(); e=expected.detach().cpu(); s=score.detach().cpu().float()
        pos=positions.detach().cpu()
        for row in range(config.T):
            if row>=n:
                if bool((a[row]!=padding).any()): return False,"inactive top-k row is not padded"
                continue
            visible=min((int(pos[row].item())+1)//config.COMPRESS_RATIO,prefix_cap)
            prefix=a[row,:visible]
            if visible:
                if bool(((prefix<0)|(prefix>=visible)).any()): return False,"top-k index out of visible range"
                if int(torch.unique(prefix).numel())!=visible: return False,"top-k visible prefix is not unique"
            if bool((a[row,visible:]!=padding).any()): return False,"top-k tail is not padded"
        # Exact selected topk_pair_compare mechanics: only index mismatch
        # positions impose ordering across their two neighboring score pairs.
        indices=a.long().clamp(min=0,max=s.shape[-1]-1)
        paired=torch.gather(s,dim=-1,index=indices)
        paired=torch.where(a<0,torch.full_like(paired,-torch.inf),paired)
        mismatch=a!=e
        pair_ok=paired[:,:-1]>=paired[:,1:]
        left_ok=torch.ones_like(mismatch); right_ok=torch.ones_like(mismatch)
        left_ok[:,1:]=pair_ok; right_ok[:,:-1]=pair_ok
        fail=mismatch & ~(left_ok & right_ok)
        return not bool(fail.any()), "top-k actual score order breaks at index mismatch" if bool(fail.any()) else ""
    compare.__name__=rule["name"]
    return compare

def make_compare(case_dir, case):
    rules=_policy(case_dir,case); config=_config(case_dir,case); callbacks={}
    for name,rule in rules.items():
        if not isinstance(rule,dict): raise ValueError("invalid comparator rule")
        callbacks[name] = _score(rule) if name=="score" else _topk(rule,config) if name=="topk_idxs" else _mapped_active_rows(rule, (config.PREFILL_IDX_BLOCK_NUM,config.BLOCK_SIZE,1,config.IDX_HEAD_DIM if name=="idx_kv_cache" else 1))
    if float(rules["score"]["tail_sentinel"]) != float(config.FP32_NEG_INF): raise ValueError("score sentinel disagrees with fixed config")
    return callbacks

def build_specs(fixture, config):
    specs=[]
    for name in _NAMES:
        if name=="num_tokens":
            value=torch.load(fixture/"in/num_tokens.pt",map_location="cpu",weights_only=True); specs.append(ScalarSpec(name,torch.int32,int(value.item()))); continue
        if name in {"score","topk_idxs"}:
            dtype = torch.float32 if name == "score" else torch.int32
            specs.append(TensorSpec(name,[config.T, config.INDEXER_SCORE_CAP],dtype)); continue
        value=torch.load(fixture/"in"/f"{name}.pt",map_location="cpu",weights_only=True)
        specs.append(TensorSpec(name,list(value.shape),value.dtype,init_value=value))
    return specs

def _program(module,case): return getattr(module,case["entry"])
def _oracle_golden(oracle,values):
    outputs=oracle.compute(values)
    if set(outputs)!={"score","idx_kv_cache","idx_kv_scale","topk_idxs"}: raise ValueError("Oracle output coverage mismatch")
    for name,value in outputs.items(): values[name].copy_(value)

def prepare_ref(case_dir,sealed,case,platform=None):
    config=_config(case_dir,case); identity=case["case_id"].replace(".","_")
    return workflow.prepare_ref(case_dir=case_dir,sealed=sealed,case=case,platform=platform,build_specs=lambda f:build_specs(f,config),load_ref=lambda:common.load_ref(case_dir,sealed,f"ref_{identity}"),load_oracle=lambda:common.load_oracle(case_dir,sealed,f"oracle_{identity}"),make_program=_program,make_compare=lambda:make_compare(case_dir,case),run=run,oracle_golden=_oracle_golden)

def evaluate_candidate(case_dir,sealed,case,candidate_workspace,platform=None,phase="all"):
    config=_config(case_dir,case); identity=case["case_id"].replace(".","_")
    return workflow.evaluate_candidate(case_dir=case_dir,sealed=sealed,case=case,candidate_workspace=candidate_workspace,platform=platform,phase=phase,build_specs=lambda f:build_specs(f,config),load_candidate=lambda:common.load_candidate(case_dir,case,candidate_workspace,f"candidate_{identity}"),make_program=_program,make_compare=lambda:make_compare(case_dir,case),run=run)
