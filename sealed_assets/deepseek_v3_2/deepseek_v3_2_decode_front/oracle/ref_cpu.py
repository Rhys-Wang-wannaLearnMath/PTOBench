"""Case-local CPU Oracle for DeepSeek V3.2 fused decode front."""

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
V_HEAD_DIM = 128
KV_A_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM
CACHE_ROWS = BATCH * MAX_SEQ
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_Q_OUT = INDEX_HEADS * INDEX_HEAD_DIM
INDEX_TOPK = 2048
EP_NODES = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
EPS = 1.0e-6
HIDDEN_INV = 1.0 / HIDDEN
WEIGHT_SCALE = (INDEX_HEADS ** -0.5) * (INDEX_HEAD_DIM ** -0.5)
FP32_NEG_INF = -3.4028234663852886e38
SORT_LEN = 8192


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"decode_front has no scalars: {sorted(scalars)}")
    hidden_states = inputs["hidden_states"].float()
    input_rms_weight = inputs["input_rms_weight"].float()
    q_norm_weight = inputs["q_norm_weight"].float()
    kv_norm_weight = inputs["kv_norm_weight"].float()
    wq_a = inputs["wq_a"].float()
    wq_b = inputs["wq_b"].float()
    wkv_a = inputs["wkv_a"].float()
    seq_lens = inputs["seq_lens"]
    if torch.any(seq_lens <= 0):
        raise ValueError("decode_front expects seq_lens >= 1")
    rope_cos = inputs["rope_cos"].float()
    rope_sin = inputs["rope_sin"].float()
    wq_b_idx = inputs["wq_b_idx"].float()
    wk_idx = inputs["wk_idx"].float()
    weights_proj = inputs["weights_proj"].float()
    k_norm_weight = inputs["k_norm_weight"].float()
    k_norm_bias = inputs["k_norm_bias"].float()
    hadamard_q = inputs["hadamard_q"].float()
    hadamard_k = inputs["hadamard_k"].float()
    layer_id = int(inputs["layer_id_t"][0].item())
    w_q_nope_to_latent = inputs["w_q_nope_to_latent"].float()
    w_latent_to_v = inputs["w_latent_to_v"].float()
    kv_cache = inputs["kv_cache"].clone()
    pe_cache = inputs["pe_cache"].clone()
    k_cache_idx = inputs["k_cache_idx"].clone()
    dispatch_buf = inputs["dispatch_buf"].clone()

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
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        cache_row = b * MAX_SEQ + pos
        cos_lo = rope_cos[pos:pos + 1, :half]
        cos_hi = rope_cos[pos:pos + 1, half:]
        sin_lo = rope_sin[pos:pos + 1, :half]
        sin_hi = rope_sin[pos:pos + 1, half:]

        q_pe = q_proj_view[b, :, QK_NOPE_HEAD_DIM:]
        q_lo = q_pe[:, :half].clone()
        q_hi = q_pe[:, half:].clone()
        q_proj_view[b, :, QK_NOPE_HEAD_DIM:QK_NOPE_HEAD_DIM + half] = q_lo * cos_lo - q_hi * sin_lo
        q_proj_view[b, :, QK_NOPE_HEAD_DIM + half:] = q_hi * cos_hi + q_lo * sin_hi

        kv_row = kv_a[b:b + 1, :KV_LORA_RANK].float()
        kv_var = torch.mean(kv_row * kv_row, dim=-1, keepdim=True)
        kv_normed = kv_row * torch.rsqrt(kv_var + EPS) * kv_norm_weight
        kv_cache[cache_row:cache_row + 1].copy_(kv_normed.to(torch.bfloat16))

        pe_lo = kv_a[b:b + 1, KV_LORA_RANK:KV_LORA_RANK + half].float()
        pe_hi = kv_a[b:b + 1, KV_LORA_RANK + half:KV_LORA_RANK + 2 * half].float()
        pe_cache[cache_row:cache_row + 1, :half].copy_((pe_lo * cos_lo - pe_hi * sin_lo).to(torch.bfloat16))
        pe_cache[cache_row:cache_row + 1, half:].copy_((pe_hi * cos_hi + pe_lo * sin_hi).to(torch.bfloat16))

    q_idx_full = (qr.float() @ wq_b_idx).to(torch.bfloat16).float()
    k_idx = (hidden_states @ wk_idx).to(torch.bfloat16).float()
    mean = k_idx.mean(dim=-1, keepdim=True)
    centered = k_idx - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    k_idx = (centered * torch.rsqrt(var + EPS) * k_norm_weight + k_norm_bias).to(torch.bfloat16).float()

    q_view = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        cos_lo = rope_cos[pos:pos + 1, :half]
        cos_hi = rope_cos[pos:pos + 1, half:QK_ROPE_HEAD_DIM]
        sin_lo = rope_sin[pos:pos + 1, :half]
        sin_hi = rope_sin[pos:pos + 1, half:QK_ROPE_HEAD_DIM]
        q_pe_i = q_view[b, :, :QK_ROPE_HEAD_DIM]
        q_lo = q_pe_i[:, :half].clone()
        q_hi = q_pe_i[:, half:].clone()
        q_view[b, :, :half] = q_lo * cos_lo - q_hi * sin_lo
        q_view[b, :, half:QK_ROPE_HEAD_DIM] = q_hi * cos_hi + q_lo * sin_hi
        k_lo = k_idx[b:b + 1, :half].clone()
        k_hi = k_idx[b:b + 1, half:QK_ROPE_HEAD_DIM].clone()
        k_idx[b:b + 1, :half] = k_lo * cos_lo - k_hi * sin_lo
        k_idx[b:b + 1, half:QK_ROPE_HEAD_DIM] = k_hi * cos_hi + k_lo * sin_hi

    q_idx_full = torch.einsum("bhd,dk->bhk", q_view, hadamard_q).reshape(BATCH, INDEX_Q_OUT).to(torch.bfloat16).float()
    k_idx = (k_idx @ hadamard_k).to(torch.bfloat16).float()
    weights = (hidden_states @ weights_proj.to(torch.bfloat16).float()) * WEIGHT_SCALE
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        k_cache_idx[b * MAX_SEQ + pos, :].copy_(k_idx[b].to(torch.bfloat16))

    q_heads = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    q_idx = torch.einsum("bhd,bh->bd", q_heads, weights)
    scores = torch.full((BATCH, SORT_LEN), FP32_NEG_INF, dtype=torch.float32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        q_b = q_idx[b:b + 1]
        k_b = k_cache_idx[b * MAX_SEQ:b * MAX_SEQ + ctx_len].float()
        scores[b, :ctx_len] = (q_b @ k_b.T).squeeze(0)
    _, idx = torch.topk(scores, INDEX_TOPK, dim=1, largest=True, sorted=True)
    idx = idx.to(torch.int32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        valid_topk = min(INDEX_TOPK, ctx_len)
        idx[b, valid_topk:] = torch.iinfo(torch.int32).min

    attn_front = torch.zeros(BATCH, ATTN_OUT, dtype=torch.float32)
    dispatch_buf.zero_()
    attn_scale = 1.0 / (QK_HEAD_DIM ** 0.5)
    for b in range(BATCH):
        sparse_k = min(INDEX_TOPK, int(seq_lens[b].item()))
        q_nope = q_proj_view[b, :, :QK_NOPE_HEAD_DIM]
        q_pe = q_proj_view[b, :, QK_NOPE_HEAD_DIM:]
        q_nope_latent = torch.empty(NUM_HEADS, KV_LORA_RANK, dtype=torch.float32)
        for h in range(NUM_HEADS):
            q_nope_latent[h].copy_((q_nope[h:h + 1] @ w_q_nope_to_latent[h]).squeeze(0))

        oi = torch.zeros(NUM_HEADS, KV_LORA_RANK, dtype=torch.float32)
        li = torch.zeros(NUM_HEADS, 1, dtype=torch.float32)
        mi = torch.zeros(NUM_HEADS, 1, dtype=torch.float32)
        initialized = False
        for kk in range(sparse_k):
            topk_pos = int(idx[b, kk].item())
            if topk_pos < 0:
                continue
            cache_s = b * MAX_SEQ + topk_pos
            kv_s = kv_cache[cache_s:cache_s + 1].float()
            pe_s = pe_cache[cache_s:cache_s + 1].float()
            score_nope = (q_nope_latent * kv_s).sum(dim=-1, keepdim=True)
            score_pe = (q_pe * pe_s).sum(dim=-1, keepdim=True)
            cur_mi = (score_nope + score_pe) * attn_scale
            if not initialized:
                oi = kv_s.expand(NUM_HEADS, KV_LORA_RANK).clone()
                li = torch.ones(NUM_HEADS, 1, dtype=torch.float32)
                mi = cur_mi.clone()
                initialized = True
            else:
                mi_new = torch.maximum(mi, cur_mi)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                li = alpha * li + beta
                oi = oi * alpha + kv_s * beta
                mi = mi_new
        ctx_latent = oi / li.clamp_min(1e-30)
        ctx_v = torch.empty(NUM_HEADS, V_HEAD_DIM, dtype=torch.float32)
        for h in range(NUM_HEADS):
            ctx_v[h].copy_((ctx_latent[h:h + 1] @ w_latent_to_v[h]).squeeze(0))
        attn_front[b].copy_(ctx_v.reshape(ATTN_OUT))

    for b in range(BATCH):
        target_node = (b + layer_id) % EP_NODES
        dispatch_buf[target_node, b].copy_(attn_front[b].to(torch.bfloat16))

    return {
        "kv_cache": kv_cache.contiguous(),
        "pe_cache": pe_cache.contiguous(),
        "k_cache_idx": k_cache_idx.contiguous(),
        "dispatch_buf": dispatch_buf.contiguous(),
    }

