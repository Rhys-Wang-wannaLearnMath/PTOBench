"""Independent CPU Oracle for DeepSeek-V4 HC prefill pre-mix."""
import torch
from config import B, S, T, D, EPS, HC_DIM, HC_EPS, HC_MULT, HC_SINKHORN_ITER
MIX_HC = (2 + HC_MULT) * HC_MULT
HD = HC_DIM
def _bf16(v):r=(v.contiguous().view(torch.int32)+0x8000)&-0x10000;return r.view(torch.float32).to(torch.bfloat16)
def compute(i, s):
 if s: raise ValueError("no scalars")
 source=i["x"]; x=source.flatten(2).reshape(T, HD).float(); w=i["hc_fn"].float(); sq=torch.zeros(T, 1)
 for k in range(0, HD, 128):
  q=x[:, k:k+128]; sq += (q*q).sum(1, keepdim=True)
 inv=torch.rsqrt(sq/HD + EPS); cols=[]
 for m in range(MIX_HC):
  dot=torch.zeros(T, 1)
  for k in range(0, HD, 512): dot += (x[:, k:k+512]*w[m:m+1, k:k+512]).sum(1, keepdim=True)
  cols.append(dot*inv)
 mix_2d=torch.cat(cols, 1); mix=mix_2d.reshape(B, S, MIX_HC); scale=i["hc_scale"].float(); base=i["hc_base"].float(); pre=torch.sigmoid(mix[..., :HC_MULT]*scale[0]+base[:HC_MULT])+HC_EPS
 # Preserve the official Golden's padded eight-column sigmoid call before
 # slicing the four logical post lanes.  Computing only four columns can differ
 # by one FP32 ULP because Torch selects a different vectorized sigmoid kernel.
 post_logits=mix_2d[:, HC_MULT:HC_MULT+HC_MULT*2]*scale[1]+base[HC_MULT:HC_MULT+HC_MULT*2]; post=(2*torch.sigmoid(post_logits[:, :HC_MULT])).reshape(B, S, HC_MULT)
 comb=(mix[..., HC_MULT*2:]*scale[2]+base[HC_MULT*2:]).view(B,S,HC_MULT,HC_MULT); comb=torch.softmax(comb,-1)+HC_EPS; comb=comb/(comb.sum(-2,keepdim=True)+HC_EPS)
 for _ in range(HC_SINKHORN_ITER-1): comb=comb/(comb.sum(-1,keepdim=True)+HC_EPS); comb=comb/(comb.sum(-2,keepdim=True)+HC_EPS)
 lanes=source.float(); y=(lanes[:,:,0]*pre[:,:,0:1]+lanes[:,:,1]*pre[:,:,1:2])+(lanes[:,:,2]*pre[:,:,2:3]+lanes[:,:,3]*pre[:,:,3:4])
 return {"x_mixed":_bf16(y),"post":post,"comb":comb}
