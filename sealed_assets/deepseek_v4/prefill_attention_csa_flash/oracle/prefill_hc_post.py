from config import CASE_CONFIG as M, PREFILL_BATCH, PREFILL_SEQ

B = PREFILL_BATCH

S = PREFILL_SEQ

D = M.hidden_size

HC_MULT = M.hc_mult

def golden_prefill_hc_post(tensors):
    import torch

    x = tensors["x"].float()
    residual = tensors["residual"].float()
    post = tensors["post"].float()
    comb = tensors["comb"].float()

    y_fp32 = torch.zeros(B, S, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, :, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, :, in_h, :] * comb[:, :, in_h, out_h:out_h + 1]
        y_fp32[:, :, out_h, :] = y_row
    tensors["y"][:] = y_fp32.to(torch.bfloat16)
