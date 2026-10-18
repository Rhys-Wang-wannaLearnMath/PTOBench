"""External B16 decode HC-post Oracle."""
import torch
from config import B, S, HC_MULT as H, D
def compute(i,s):
 if s: raise ValueError("no scalars")
 x=i["x"].float(); r=i["residual"].float(); p=i["post"].float(); c=i["comb"].float(); y=torch.zeros(B,S,H,D)
 for o in range(H):
  row=x*p[:,:,o:o+1]
  for h in range(H): row=row+r[:,:,h]*c[:,:,h,o:o+1]
  y[:,:,o]=row
 return {"y":y.to(torch.bfloat16)}
