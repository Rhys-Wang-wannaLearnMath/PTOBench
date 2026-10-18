"""DeepSeek-V4 sliding-window decode implementation."""

import pypto.language as pl

from config import CASE_CONFIG as M, BLOCK_SIZE, DECODE_BATCH, DECODE_SEQ
from decode_qkv_proj_rope import attn_norm, qkv_proj_rope
from decode_sparse_attn import sparse_attn
from hc_post import hc_post
from hc_pre import hc_pre


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# Implementation-local topology and tiling constants.
ORI_MAX_BLOCKS = 1
TOPK = WIN
SPARSE_IDX_TOPK = M.index_topk
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK
SPARSE_CMP_MAX_BLOCKS = 64
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE


@pl.jit.inline
def attention_swa(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post_t = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_mixed = hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post_t,
        comb_t,
    )

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_step"):
        for b in pl.parallel(B):
            start_pos_b = pl.read(start_pos, [b])
            for s_idx in pl.range(S):
                pos_b = pl.cast(start_pos_b + s_idx, pl.INDEX)
                cos_row = pl.cast(
                    pl.slice(freqs_cos, [1, ROPE_HEAD_DIM], [pos_b, 0]),
                    target_type=pl.FP32,
                )
                sin_row = pl.cast(
                    pl.slice(freqs_sin, [1, ROPE_HEAD_DIM], [pos_b, 0]),
                    target_type=pl.FP32,
                )
                t = b * S + s_idx
                rope_cos_t = pl.assemble(
                    rope_cos_t,
                    pl.cast(cos_row, target_type=pl.BF16, mode="rint"),
                    [t, 0],
                )
                rope_sin_t = pl.assemble(
                    rope_sin_t,
                    pl.cast(sin_row, target_type=pl.BF16, mode="rint"),
                    [t, 0],
                )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    x_normed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    x_normed = attn_norm(x_mixed, attn_norm_w, x_normed)
    q = qkv_proj_rope(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos_t,
        rope_sin_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )

    kv_cache_flat = pl.reshape(
        kv_cache, [B * ORI_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM]
    )
    sparse_topk = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    cmp_kv_dummy = pl.create_tensor(
        [B * SPARSE_CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16
    )
    cmp_block_table_dummy = pl.create_tensor(
        [B, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_post_qkv"):
        cmp_block_table_dummy[:, :] = pl.full(
            [B, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32, value=-1
        )
        idx_row = pl.arange(0, [1, WIN], dtype=pl.INT32)
        pad_row = pl.full([1, SPARSE_IDX_TOPK], dtype=pl.INT32, value=-1)
        sparse_topk_row = pl.concat(idx_row, pad_row)
        for t in pl.range(T):
            sparse_topk[t : t + 1, :] = sparse_topk_row
        for b in pl.range(B):
            start_pos_b = pl.read(start_pos, [b])
            for s_idx in pl.range(S):
                ori_slot = (start_pos_b + s_idx) % WIN
                blk_id = pl.cast(
                    pl.read(block_table, [b, ori_slot // BLOCK_SIZE]), pl.INDEX
                )
                dst_row = blk_id * BLOCK_SIZE + ori_slot % BLOCK_SIZE
                kv_cache_flat[dst_row : dst_row + 1, :] = kv[
                    b * S + s_idx : b * S + s_idx + 1, :
                ]
    kv_cache = pl.reshape(
        kv_cache_flat, [B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM]
    )

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn(
        q,
        kv_cache,
        block_table,
        cmp_kv_dummy,
        cmp_block_table_dummy,
        sparse_topk,
        attn_sink,
        seqused_kv,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )

    attn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out = hc_post(attn_out_3d, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def attention_swa_test(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_out = attention_swa(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        kv_cache,
        block_table,
        attn_sink,
        seqused_kv,
        wo_a,
        wo_b,
        wo_b_scale,
        x_out,
        start_pos,
    )
    return x_out
