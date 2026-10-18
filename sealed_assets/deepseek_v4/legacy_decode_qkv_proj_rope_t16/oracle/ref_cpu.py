"""External Oracle for the T=16 legacy decode QKV/RoPE program."""

import torch

from config import (
    D,
    EPS,
    H,
    HEAD_DIM as HD,
    INT8_AMAX_EPS as FLOOR,
    INT8_SCALE_MAX as MAX,
    NOPE_DIM as NOPE,
    Q_LORA as LORA,
    ROPE_HALF as HALF,
    T,
)


def _canonical_selectors(inputs):
    even=torch.zeros((HALF,64),dtype=torch.bfloat16);odd=torch.zeros_like(even);idx=torch.arange(HALF)
    even[idx,2*idx]=1;odd[idx,2*idx+1]=1
    if not torch.equal(inputs["even_select_t"],even) or not torch.equal(inputs["odd_select_t"],odd):
        raise ValueError("legacy decode QKV requires canonical even/odd RoPE selection matrices")


def _away(value):
    return torch.sign(value)*torch.floor(torch.abs(value)+0.5)


def _rms(value,gamma):
    return value*torch.rsqrt(value.square().mean(-1,keepdim=True)+EPS)*gamma


def _mm(left,right):
    return left.to(torch.bfloat16).float().matmul(right.to(torch.bfloat16).float())


def _quant(value):
    rows=value.reshape(-1,value.shape[-1]).float();amax=rows.abs().amax(-1,keepdim=True).clamp_min(FLOOR);mult=MAX/amax
    q=_away(rows*mult).to(torch.int32).to(torch.float16).to(torch.int8).reshape_as(value)
    return q,(1.0/mult).reshape(*value.shape[:-1],1)


def _rope(value,inputs):
    pair=value.unflatten(-1,(-1,2));even,odd=pair[...,0],pair[...,1]
    cos=inputs["rope_cos"].float()[:,:HALF];sin=inputs["rope_sin"].float()[:,:HALF]
    while cos.ndim<even.ndim:cos=cos.unsqueeze(-2);sin=sin.unsqueeze(-2)
    re=(even*cos-odd*sin).to(torch.bfloat16);ro=(even*sin+odd*cos).to(torch.bfloat16)
    return torch.stack((re,ro),-1).flatten(-2)


def compute(inputs,scalars):
    if scalars:raise ValueError(f"legacy decode QKV has no scalars: {sorted(scalars)}")
    _canonical_selectors(inputs)
    token=_rms(inputs["x"].float().view(T,D),inputs["norm_w"].float())
    lora=_rms(_mm(token,inputs["wq_a"]),inputs["gamma_cq"].float())
    qr,qr_scale=_quant(lora.to(torch.bfloat16).float())
    q_int=qr.to(torch.int32).matmul(inputs["wq_b"].to(torch.int32))
    q_full=(q_int.float()*qr_scale*inputs["wq_b_scale"].float().reshape(1,-1)).reshape(T,H,HD)
    q_full=q_full*torch.rsqrt(q_full.square().mean(-1,keepdim=True)+EPS)
    q=torch.cat((q_full[...,:NOPE],_rope(q_full[...,NOPE:],inputs)),-1)
    kv_full=_rms(_mm(token,inputs["wkv"]),inputs["gamma_ckv"].float())
    kv=torch.cat((kv_full[...,:NOPE],_rope(kv_full[...,NOPE:].unsqueeze(1),inputs).squeeze(1)),-1)
    return {"q":q.to(torch.bfloat16),"kv":kv.to(torch.bfloat16),"qr":qr,"qr_scale":qr_scale}

