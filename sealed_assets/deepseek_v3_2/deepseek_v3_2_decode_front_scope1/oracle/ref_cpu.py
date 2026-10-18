"""Case-local CPU Oracle for DeepSeek V3.2 decode front scope1."""

import torch

BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
NUM_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
KV_A_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM
EPS = 1.0e-6
HIDDEN_INV = 1.0 / HIDDEN


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"decode_front_scope1 has no scalars: {sorted(scalars)}")
    hidden_states = inputs["hidden_states"].float()
    input_rms_weight = inputs["input_rms_weight"].float()
    wq_a = inputs["wq_a"].float()
    q_norm_weight = inputs["q_norm_weight"].float()
    wq_b = inputs["wq_b"].float()
    wkv_a = inputs["wkv_a"].float()
    seq_lens = inputs["seq_lens"]
    rope_cos = inputs["rope_cos"].float()
    rope_sin = inputs["rope_sin"].float()
    kv_norm_weight = inputs["kv_norm_weight"].float()
    kv_cache = inputs["kv_cache"].clone()
    pe_cache = inputs["pe_cache"].clone()

    sq_sum = torch.sum(hidden_states * hidden_states, dim=1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * HIDDEN_INV + EPS)
    normed = (hidden_states * inv_rms * input_rms_weight).to(torch.bfloat16).float()

    qr_raw = (normed @ wq_a).to(torch.bfloat16)
    qr_raw_fp32 = qr_raw.float()
    q_var = torch.mean(qr_raw_fp32 * qr_raw_fp32, dim=1, keepdim=True)
    qr = (qr_raw_fp32 * torch.rsqrt(q_var + EPS) * q_norm_weight).to(torch.bfloat16)
    q_proj = (qr.float() @ wq_b).to(torch.bfloat16)
    kv_a = (normed @ wkv_a).to(torch.bfloat16)

    half = QK_ROPE_HEAD_DIM // 2
    q_proj_view = q_proj.float().view(q_proj.shape[0], NUM_HEADS, QK_HEAD_DIM)
    for b in range(kv_a.shape[0]):
        ctx_len = int(seq_lens[b].item())
        pos = ctx_len - 1
        cache_row = b * MAX_SEQ + pos

        kv_row = kv_a[b:b + 1, :KV_LORA_RANK].float()
        kv_var = torch.mean(kv_row * kv_row, dim=-1, keepdim=True)
        kv_normed = kv_row * torch.rsqrt(kv_var + EPS) * kv_norm_weight
        kv_cache[cache_row:cache_row + 1].copy_(kv_normed.to(torch.bfloat16))

        pe_lo = kv_a[b:b + 1, KV_LORA_RANK:KV_LORA_RANK + half].float()
        pe_hi = kv_a[b:b + 1, KV_LORA_RANK + half:KV_LORA_RANK + 2 * half].float()
        cos_lo = rope_cos[pos:pos + 1, :half]
        cos_hi = rope_cos[pos:pos + 1, half:]
        sin_lo = rope_sin[pos:pos + 1, :half]
        sin_hi = rope_sin[pos:pos + 1, half:]

        q_pe = q_proj_view[b, :, QK_NOPE_HEAD_DIM:]
        q_lo = q_pe[:, :half].clone()
        q_hi = q_pe[:, half:].clone()
        q_proj_view[b, :, QK_NOPE_HEAD_DIM:QK_NOPE_HEAD_DIM + half] = q_lo * cos_lo - q_hi * sin_lo
        q_proj_view[b, :, QK_NOPE_HEAD_DIM + half:] = q_hi * cos_hi + q_lo * sin_hi
        pe_cache[cache_row:cache_row + 1, :half].copy_((pe_lo * cos_lo - pe_hi * sin_lo).to(torch.bfloat16))
        pe_cache[cache_row:cache_row + 1, half:].copy_((pe_hi * cos_hi + pe_lo * sin_hi).to(torch.bfloat16))

    return {
        "qr_out": qr.contiguous(),
        "q_nope_out": q_proj_view[:, :, :QK_NOPE_HEAD_DIM].reshape(q_proj.shape[0], -1).to(torch.bfloat16).contiguous(),
        "q_pe_out": q_proj_view[:, :, QK_NOPE_HEAD_DIM:].reshape(q_proj.shape[0], -1).to(torch.bfloat16).contiguous(),
        "kv_a_out": kv_a.contiguous(),
        "kv_cache": kv_cache.contiguous(),
        "pe_cache": pe_cache.contiguous(),
    }

