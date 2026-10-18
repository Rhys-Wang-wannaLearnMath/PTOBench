"""Case-local CPU Oracle for DeepSeek V3.2 decode front scope4 full profile."""

import torch
from config import EP_NODES


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"decode_front_scope4 has no scalars: {sorted(scalars)}")
    q_proj = inputs["q_proj"].float()
    kv_cache = inputs["kv_cache"].float()
    pe_cache = inputs["pe_cache"].float()
    topk_idx = inputs["topk_idx"]
    seq_lens = inputs["seq_lens"]
    layer_id = int(inputs["layer_id_t"][0].item())
    w_q_nope_to_latent = inputs["w_q_nope_to_latent"].float()
    w_latent_to_v = inputs["w_latent_to_v"].float()

    batch = q_proj.shape[0]
    num_heads = w_q_nope_to_latent.shape[0]
    kv_lora_rank = w_q_nope_to_latent.shape[2]
    qk_rope_head_dim = pe_cache.shape[1]
    qk_head_dim = q_proj.shape[1] // num_heads
    qk_nope_head_dim = qk_head_dim - qk_rope_head_dim
    v_head_dim = w_latent_to_v.shape[2]
    attn_out = num_heads * v_head_dim
    index_topk = topk_idx.shape[1]
    max_seq = kv_cache.shape[0] // batch
    ep_nodes = EP_NODES
    attn_scale = 1.0 / (qk_head_dim ** 0.5)

    attn_front = torch.zeros(batch, attn_out, dtype=torch.float32)
    dispatch_buf = torch.zeros(ep_nodes, batch, attn_out, dtype=torch.bfloat16)
    for b in range(batch):
        sparse_k = min(index_topk, int(seq_lens[b].item()))
        q_proj_b = q_proj[b].view(num_heads, qk_head_dim)
        q_nope = q_proj_b[:, :qk_nope_head_dim]
        q_pe = q_proj_b[:, qk_nope_head_dim:qk_head_dim]
        q_nope_latent = torch.empty(num_heads, kv_lora_rank, dtype=torch.float32)
        for h in range(num_heads):
            q_nope_latent[h].copy_((q_nope[h:h + 1] @ w_q_nope_to_latent[h]).squeeze(0))

        oi = torch.zeros(num_heads, kv_lora_rank, dtype=torch.float32)
        li = torch.zeros(num_heads, 1, dtype=torch.float32)
        mi = torch.zeros(num_heads, 1, dtype=torch.float32)
        initialized = False
        for kk in range(sparse_k):
            topk_pos = int(topk_idx[b, kk].item())
            if topk_pos < 0:
                continue
            cache_s = b * max_seq + topk_pos
            kv_s = kv_cache[cache_s:cache_s + 1]
            pe_s = pe_cache[cache_s:cache_s + 1]
            score_nope = (q_nope_latent * kv_s).sum(dim=-1, keepdim=True)
            score_pe = (q_pe * pe_s).sum(dim=-1, keepdim=True)
            cur_mi = (score_nope + score_pe) * attn_scale
            if not initialized:
                oi = kv_s.expand(num_heads, kv_lora_rank).clone()
                li = torch.ones(num_heads, 1, dtype=torch.float32)
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
        ctx_v = torch.empty(num_heads, v_head_dim, dtype=torch.float32)
        for h in range(num_heads):
            ctx_v[h].copy_((ctx_latent[h:h + 1].to(torch.bfloat16).float() @ w_latent_to_v[h]).squeeze(0))
        attn_front[b].copy_(ctx_v.reshape(attn_out))

    for b in range(batch):
        target_node = (b + layer_id) % ep_nodes
        dispatch_buf[target_node, b].copy_(attn_front[b].to(torch.bfloat16))
    return {"dispatch_buf": dispatch_buf.contiguous()}
