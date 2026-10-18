"""Full fixed QKV-v2 ABI and original ratio-allclose comparison."""
from pathlib import Path
import math,torch
from golden import TensorSpec,run
import common
ORDER=('x','wq_a','wq_b','wq_b_scale','wkv','rope_cos','rope_sin','gamma_cq','gamma_ckv','q','kv','qr','qr_scale')
OUTPUTS={'q','kv','qr','qr_scale'}
INPUTS=set(ORDER)-OUTPUTS

def build_tensor_specs(fixture:Path,config=None):
    values={p.stem:torch.load(p,map_location='cpu',weights_only=True) for p in (fixture/'in').glob('*.pt')}
    if set(values)!=INPUTS:raise ValueError('fixed Fixture coverage mismatch')
    t,d=values['x'].shape;ql=values['gamma_cq'].numel();hd=values['gamma_ckv'].numel();h=values['wq_b_scale'].numel()//hd;r=values['rope_cos'].shape[1]
    if config is not None and (t,d,ql,hd,h,r)!=(config.T,config.D,config.Q_LORA,config.HEAD_DIM,config.H,config.ROPE_DIM):raise ValueError('Fixture fixed config mismatch')
    shapes={'x':(t,d),'wq_a':(d,ql),'wq_b':(ql,h*hd),'wq_b_scale':(h*hd,),'wkv':(d,hd),'rope_cos':(t,r),'rope_sin':(t,r),'gamma_cq':(ql,),'gamma_ckv':(hd,),'q':(t,h,hd),'kv':(t,hd),'qr':(t,ql),'qr_scale':(t,1)}
    dtypes={n:torch.bfloat16 for n in ORDER};dtypes.update(wq_b=torch.int8,wq_b_scale=torch.float32,qr=torch.int8,qr_scale=torch.float32)
    specs=[]
    for n in ORDER:
        v=values.get(n)
        if v is not None and (v.shape!=shapes[n] or v.dtype!=dtypes[n]):raise ValueError('Fixture shape/dtype mismatch: '+n)
        specs.append(TensorSpec(n,list(shapes[n]),dtypes[n],init_value=v))
    return specs

def _number(v,name):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)) or v<0:raise ValueError('invalid '+name)
    return float(v)

def _callbacks(policy,names):
    if not isinstance(policy,dict) or set(policy)!={'schema_version','outputs'} or type(policy['schema_version']) is not int or policy['schema_version']!=1:raise ValueError('invalid comparator schema')
    if not isinstance(policy['outputs'],dict) or set(policy['outputs'])!=set(names):raise ValueError('output coverage mismatch')
    callbacks={}
    for n in names:
        rule=policy['outputs'][n]
        if not isinstance(rule,dict) or set(rule)!={'method','rtol','atol','max_error_ratio'} or rule['method']!='ratio_allclose':raise ValueError('invalid ratio rule')
        rt,at,ratio=(_number(rule[k],n+'.'+k) for k in ('rtol','atol','max_error_ratio'))
        if ratio>1:raise ValueError('invalid error ratio')
        def compare(actual,expected,*,policy_rtol=rt,policy_atol=at,policy_ratio=ratio,**kwargs):
            if actual.shape!=expected.shape or actual.dtype!=expected.dtype:return False,'shape or dtype mismatch'
            actual_f=actual.to(torch.float32);expected_f=expected.to(torch.float32)
            if not bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all()):return False,'non-finite output'
            mismatch_count=int(((actual_f-expected_f).abs()>policy_atol+policy_rtol*expected_f.abs()).sum().item())
            allowed=round(policy_ratio*actual_f.numel())
            return mismatch_count<=allowed,f'{mismatch_count}/{actual_f.numel()} elements exceed tolerance; allowed {allowed}'
        callbacks[n]=compare
    return callbacks

def _compare(case_dir,case):return _callbacks(common.read_json(case_dir/case['comparator']),case['correctness']['outputs'])
def _program(module,case):return getattr(module,case['entry'])
def _oracle(oracle,values):
    for n,v in oracle.compute({n:values[n] for n in INPUTS},{}).items():values[n][:]=v

def prepare_ref(case_dir,sealed,case,platform=None):
    identity=case['case_id'].replace('.','_');workflow=common.load_module(common.BENCHMARK_ROOT/'evaluator/workflow.py',identity+'_workflow')
    cfg=common.load_module(case_dir/case['config'],identity+'_evaluator_config')
    return workflow.prepare_ref(case_dir=case_dir,sealed=sealed,case=case,platform=platform,build_specs=lambda fixture:build_tensor_specs(fixture,cfg),load_ref=lambda:common.load_ref(case_dir,sealed,identity+'_ref'),load_oracle=lambda:common.load_oracle(case_dir,sealed,identity+'_oracle'),make_program=_program,make_compare=lambda:_compare(case_dir,case),run=run,oracle_golden=_oracle)

def evaluate_candidate(case_dir,sealed,case,candidate_workspace,platform=None,phase='all'):
    identity=case['case_id'].replace('.','_');workflow=common.load_module(common.BENCHMARK_ROOT/'evaluator/workflow.py',identity+'_workflow')
    cfg=common.load_module(case_dir/case['config'],identity+'_evaluator_config')
    return workflow.evaluate_candidate(case_dir=case_dir,sealed=sealed,case=case,candidate_workspace=candidate_workspace,platform=platform,phase=phase,build_specs=lambda fixture:build_tensor_specs(fixture,cfg),load_candidate=lambda:common.load_candidate(case_dir,case,candidate_workspace,identity+'_candidate'),make_program=_program,make_compare=lambda:_compare(case_dir,case),run=run)
