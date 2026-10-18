"""External B16 class-program HC-pre Oracle."""
import torch
B,S,H,D,MIX=16,1,4,4096,24;EPS=1e-6
def compute(i,s):
 if s:raise ValueError("no scalars")
 x=i["x"].float();flat=x.flatten(2);mix=(flat@i["hc_fn"].float().T)*torch.rsqrt(flat.square().mean(-1,keepdim=True)+EPS);scale=i["hc_scale"].float();base=i["hc_base"].float();pre=torch.sigmoid(mix[...,:4]*scale[0]+base[:4])+EPS;post=2*torch.sigmoid(mix[...,4:8]*scale[1]+base[4:8]);comb=(mix[...,8:]*scale[2]+base[8:]).view(B,S,4,4);comb=torch.softmax(comb,-1)+EPS;comb=comb/(comb.sum(-2,keepdim=True)+EPS)
 for _ in range(19):comb=comb/(comb.sum(-1,keepdim=True)+EPS);comb=comb/(comb.sum(-2,keepdim=True)+EPS)
 return {"x_mixed":(pre.unsqueeze(-1)*x).sum(2).to(torch.bfloat16),"post":post,"comb":comb}

