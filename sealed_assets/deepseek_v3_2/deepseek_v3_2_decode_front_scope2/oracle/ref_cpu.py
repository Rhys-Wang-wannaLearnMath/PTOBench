"""Case-local CPU Oracle for DeepSeek V3.2 decode front scope2."""

import torch

BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
Q_LORA_RANK = 1536
QK_ROPE_HEAD_DIM = 64
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
EPS = 1.0e-6
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1.0e-4
WEIGHT_SCALE = (INDEX_HEADS ** -0.5) * (INDEX_HEAD_DIM ** -0.5)


def _int8_quant_groups(x):
    rows = x.reshape(-1, INDEX_HEAD_DIM).float()
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i32 = scaled.round().to(torch.int32)
    out_half = out_i32.to(torch.float16)
    out_i8 = out_half.to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"decode_front_scope2 has no scalars: {sorted(scalars)}")
    hidden_states = inputs["hidden_states"].float()
    qr = inputs["qr"].float()
    wq_b_idx = inputs["wq_b_idx"].float()
    wk_idx = inputs["wk_idx"].float()
    k_norm_weight = inputs["k_norm_weight"].float()
    k_norm_bias = inputs["k_norm_bias"].float()
    weights_proj = inputs["weights_proj"].float()
    seq_lens = inputs["seq_lens"]
    rope_cos = inputs["rope_cos"].float()
    rope_sin = inputs["rope_sin"].float()
    k_cache_idx_i8 = inputs["k_cache_idx_i8"].clone()
    k_cache_idx_scale = inputs["k_cache_idx_scale"].clone()

    q_idx_full = (qr @ wq_b_idx).to(torch.bfloat16).float()
    k_idx = (hidden_states @ wk_idx).to(torch.bfloat16).float()

    mean = k_idx.mean(dim=-1, keepdim=True)
    centered = k_idx - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    k_idx = (centered * torch.rsqrt(var + EPS) * k_norm_weight + k_norm_bias).to(torch.bfloat16).float()

    half = QK_ROPE_HEAD_DIM // 2
    q_view = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        cos_lo = rope_cos[pos:pos + 1, :half]
        cos_hi = rope_cos[pos:pos + 1, half:QK_ROPE_HEAD_DIM]
        sin_lo = rope_sin[pos:pos + 1, :half]
        sin_hi = rope_sin[pos:pos + 1, half:QK_ROPE_HEAD_DIM]

        q_pe = q_view[b, :, :QK_ROPE_HEAD_DIM]
        q_lo = q_pe[:, :half].clone()
        q_hi = q_pe[:, half:].clone()
        q_view[b, :, :half] = q_lo * cos_lo - q_hi * sin_lo
        q_view[b, :, half:QK_ROPE_HEAD_DIM] = q_hi * cos_hi + q_lo * sin_hi

        k_lo = k_idx[b:b + 1, :half].clone()
        k_hi = k_idx[b:b + 1, half:QK_ROPE_HEAD_DIM].clone()
        k_idx[b:b + 1, :half] = k_lo * cos_lo - k_hi * sin_lo
        k_idx[b:b + 1, half:QK_ROPE_HEAD_DIM] = k_hi * cos_hi + k_lo * sin_hi
    q_idx_full = q_view.reshape(BATCH, INDEX_HEADS * INDEX_HEAD_DIM)

    weights = (hidden_states @ weights_proj.to(torch.bfloat16).float()) * WEIGHT_SCALE
    q_idx_full = q_idx_full.to(torch.bfloat16).float()
    k_idx = k_idx.to(torch.bfloat16).float()
    q_idx_i8_golden, q_idx_scale_golden = _int8_quant_groups(q_idx_full)
    k_idx_i8_golden, k_idx_scale_golden = _int8_quant_groups(k_idx)

    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        k_cache_idx_i8[b * MAX_SEQ + pos, :].copy_(k_idx_i8_golden[b])
        k_cache_idx_scale[b, pos].copy_(k_idx_scale_golden[b, 0])

    return {
        "k_cache_idx_i8": k_cache_idx_i8.contiguous(),
        "k_cache_idx_scale": k_cache_idx_scale.contiguous(),
        "q_idx_full_i8_out": q_idx_i8_golden.view(BATCH * INDEX_HEADS, INDEX_HEAD_DIM).contiguous(),
        "q_idx_scale_heads_out": q_idx_scale_golden.view(BATCH, INDEX_HEADS).contiguous(),
        "weights_out": weights.contiguous(),
    }

