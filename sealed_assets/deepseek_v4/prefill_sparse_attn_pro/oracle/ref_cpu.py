"""Case-local CPU Oracle for Batch 08 DeepSeek V4 prefill attention."""

from __future__ import annotations

import torch

from config import BLOCK_SIZE, CASE_CONFIG as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
IDX_TOPK = M.index_topk
WIN = M.sliding_window
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 0
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, S // 4))
PREFILL_SPARSE_TOPK = min(TOPK, min(M.sliding_window, S) + PREFILL_MAX_COMPRESSED)
ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS
ROPE_CHUNK = 16
ROPE_INTERLEAVE_CHUNK = 2 * ROPE_CHUNK
GATHER_TOKEN_TILE = 4
ATTN_TOKEN_TILE = 32
ROPE_TOKEN_TILE = 8
ROPE_PACK_TOKEN_TILE = 16
MATMUL_ROW_PAD = 16
PV_HEAD_TILE = 16
MERGE_NORM_TOKEN_TILE = 16
PREFILL_ATTN_TILE = 64
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
assert 1 <= PREFILL_ATTN_BLOCKS <= 3, (
    f"PREFILL_ATTN_BLOCKS={PREFILL_ATTN_BLOCKS} from PREFILL_ATTN_TILE={PREFILL_ATTN_TILE} "
    "must fit the 3-way merge buffers"
)
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
ROPE_PACK_SPMD_BLOCKS = ((T + ROPE_PACK_TOKEN_TILE - 1) // ROPE_PACK_TOKEN_TILE) * O_GROUPS
A_K_CHUNK = 128
A_N_CHUNK = 128
B_K_CHUNK = 128
B_N_CHUNK = 128 if T >= 128 else 256
QUANT_CHUNK = 128 if T >= 128 else (128 if T >= 64 else 256)
QUANT_TOKEN_TILE = 32
PROJ_TOKEN_TILE = 128 if T >= 128 else T

def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)

def golden_prefill_sparse_attn(tensors):
    """Torch reference for the first standalone prefill sparse-attn contract."""
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors.get("cmp_kv")
    cmp_block_table = tensors.get("cmp_block_table")
    cmp_sparse_indices = tensors.get("cmp_sparse_indices")
    attn_sink = tensors["attn_sink"].float()
    seqused_kv = tensors.get("seqused_kv")
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(T):
        b = t // S
        s = t - b * S
        seq_len = int(seqused_kv[b].item()) if seqused_kv is not None else S
        if s >= seq_len:
            continue

        gathered = []
        if cmp_sparse_indices is None:
            for raw in range(min(s + 1, seq_len)):
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[blk_id, intra, 0])
        else:
            for raw_i in cmp_sparse_indices[t, :PREFILL_SPARSE_TOPK].tolist():
                raw = int(raw_i)
                if raw < 0:
                    continue
                if raw < S:
                    if raw >= seq_len:
                        continue
                    blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                    intra = raw % BLOCK_SIZE
                    gathered.append(ori_kv[blk_id, intra, 0])
                else:
                    cmp_slot = raw - S
                    if cmp_kv is None or cmp_block_table is None:
                        continue
                    if cmp_slot >= CMP_BLOCK_NUM * BLOCK_SIZE:
                        continue
                    blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
                    intra = cmp_slot % BLOCK_SIZE
                    gathered.append(cmp_kv.float()[blk_id, intra, 0])

        if not gathered:
            continue
        kv_b = torch.stack(gathered, dim=0)

        mi = None
        li = None
        oi = None
        for tile_start in range(0, kv_b.shape[0], PREFILL_ATTN_TILE):
            kv_tile = kv_b[tile_start : tile_start + PREFILL_ATTN_TILE]
            scores = (q[t] @ kv_tile.T) * SOFTMAX_SCALE
            cur_mi = scores.max(dim=-1, keepdim=True).values
            exp_scores_bf16 = torch.exp(scores - cur_mi).to(torch.bfloat16)
            cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
            cur_oi = exp_scores_bf16.float() @ kv_tile.to(torch.bfloat16).float()
            if mi is None:
                mi = cur_mi
                li = cur_li
                oi = cur_oi
            else:
                mi_new = torch.maximum(mi, cur_mi)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                li = alpha * li + beta * cur_li
                oi = oi * alpha + cur_oi * beta
                mi = mi_new

        if mi is not None:
            denom = li + torch.exp(attn_sink.unsqueeze(-1) - mi)
            o[t] = oi / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(B, S, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)

    tensors["attn_out"][:] = out.to(torch.bfloat16)

def compute(inputs, scalars):
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors.update(dict(scalars))
    tensors["attn_out"] = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn(tensors)
    return {"attn_out": tensors["attn_out"].detach().cpu().contiguous()}
