from config import CASE_CONFIG as M, PREFILL_BATCH, PREFILL_SEQ

B = PREFILL_BATCH

S = PREFILL_SEQ

T = B * S

EPS = M.rms_norm_eps

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim

HC_DIM_INV = 1.0 / HC_DIM

HC_SINKHORN_ITER = M.hc_sinkhorn_iters

HC_EPS = M.hc_eps

HC_PAD = 8

RMS_K_CHUNK = 128

LINEAR_K_CHUNK = 512

def golden_prefill_hc_pre(tensors):
    import torch

    x = tensors["x"].float()
    hc_fn = tensors["hc_fn"].float()
    hc_scale = tensors["hc_scale"].float()
    hc_base = tensors["hc_base"].float()

    x_flat = x.flatten(2)
    x_flat_2d = x_flat.reshape(T, HC_DIM)
    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)

    mix_cols = []
    for m in range(MIX_HC):
        mix_col = torch.zeros(T, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_CHUNK):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_CHUNK]
            w_chunk = hc_fn[m:m + 1, k0:k0 + LINEAR_K_CHUNK]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes_2d = torch.cat(mix_cols, dim=1)
    mixes = mixes_2d.reshape(B, S, MIX_HC)

    pre_logits = mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]
    pre = torch.sigmoid(pre_logits) + HC_EPS

    post_logits = (
        mixes_2d[:, HC_MULT:HC_MULT + HC_PAD] * hc_scale[1]
        + hc_base[HC_MULT:HC_MULT + HC_PAD]
    )
    post = (2 * torch.sigmoid(post_logits[:, :HC_MULT])).reshape(B, S, HC_MULT)

    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]).view(
        B, S, HC_MULT, HC_MULT
    )
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y01 = x[:, :, 0, :] * pre[:, :, 0:1] + x[:, :, 1, :] * pre[:, :, 1:2]
    y23 = x[:, :, 2, :] * pre[:, :, 2:3] + x[:, :, 3, :] * pre[:, :, 3:4]
    y = y01 + y23

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["x_mixed"][:] = _to_device_bf16(y)
    tensors["post"][:] = post
    tensors["comb"][:] = comb_t
