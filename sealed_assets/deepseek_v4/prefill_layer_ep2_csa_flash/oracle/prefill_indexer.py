# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill indexer.

This module builds the compressed index KV cache and per-token compressed top-k
indices consumed by packed CSA prefill sparse attention.
"""


from config import (
    CASE_CONFIG as M,
    BLOCK_SIZE,
    FP32_NEG_INF,
    IDX_CACHE_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    PREFILL_IDX_BLOCK_NUM,
)
from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    golden_prefill_indexer_compressor,
)

# model config (mirrors decode_indexer)
D = M.hidden_size
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
Q_LORA = M.q_lora_rank
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window

# kernel-local
COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
CACHE_TILE = 32

# Index cache table width mirrors decode. The physical idx_kv_cache pool is
# sized separately by PREFILL_IDX_BLOCK_NUM; keep the current score output cap
# at 256 rows because prefill_idx_score_out materializes [T, INDEXER_SCORE_CAP]
# in one Vec scope.
SPARSE_CMP_MAX_BLOCKS = 8
INDEXER_SCORE_MAX_BLOCKS = 2

B = 1
S = 128
T = B * S
START_POS = 0
TOPK_TILE = 16
assert T % TOPK_TILE == 0
INDEXER_SCORE_CAP = INDEXER_SCORE_MAX_BLOCKS * BLOCK_SIZE
assert INDEXER_SCORE_CAP == 256, "INDEXER_SCORE_CAP must stay at 256 rows"
INDEXER_SCORE_BLOCKS = max(1, (INDEXER_SCORE_CAP + CACHE_TILE - 1) // CACHE_TILE)
INDEXER_TOPK_CAP = min(IDX_TOPK, INDEXER_SCORE_CAP)
INDEXER_OFFSET = WIN + T
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)

# Q-projection / score tiling (mirrors decode_indexer)
Q_TILE = 128
Q_OUT_TILE = 256
QR_PROJ_ROW_TILE = 16
HEAD_DIM_TILE = 32
D_TILE = 32
WEIGHTS_ROW_TILE = 32
QH_QUANT_BLOCK = 256
QH_QUANT_ROW_TILE = 64
ROPE_ROW_BLOCK = IDX_N_HEADS          # one token owns IDX_N_HEADS contiguous q rows + one cos/sin
ROPE_ROW_TILE = 32
# Per-token sort-tile width. The sort32/mrgsort/gather path requires a wide tile: a narrow (256)
# sort faults on device (507018) even with a proper prefix. 2048 matches the indexer KV length and
# is the confirmed fault-free width. The real score occupies only the first INDEXER_SCORE_CAP
# columns; the rest stays -inf.
SORT_LEN = 2048
MRG_TOPK_RUN = 1024   # final mrgsort run length (>= IDX_TOPK so the top-IDX_TOPK land sorted in run 0)
# topk_pairs (= 2*PREFILL_TOPK_CAP) must be a power of two aligned to the final mrgsort run: a
# misaligned prefix (e.g. 2*192) faults like a narrow sort. valid_topk then clamps to the budget.
PREFILL_TOPK_CAP = IDX_TOPK
assert PREFILL_TOPK_CAP < SORT_LEN and SORT_LEN >= INDEXER_SCORE_CAP
SCORE_INIT_TILE = 16                   # rows per -inf init write (keep [tile, SORT_LEN] under the Vec-buffer limit)
assert T % SCORE_INIT_TILE == 0
assert (IDX_N_HEADS * IDX_HEAD_DIM) % Q_OUT_TILE == 0
assert (T * IDX_N_HEADS) % QH_QUANT_BLOCK == 0
assert ROPE_ROW_BLOCK % ROPE_ROW_TILE == 0




def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path.

    Mirrors decode_indexer._int8_quant_per_row: round-to-int8 via fp16 to match the device
    rounding, return the dequant scale (1/scale_quant) per row.
    """
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    out_i8 = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def golden_prefill_indexer_core(tensors):
    import torch

    compressor_tensors = {
        "x": tensors["x"],
        "kv": torch.zeros(MAX_CMP_WRITES, IDX_HEAD_DIM, dtype=torch.bfloat16),
        "kv_state": tensors["inner_kv_state"],
        "score_state": tensors["inner_score_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "hadamard": tensors["hadamard"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "idx_block_table": tensors["idx_block_table"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    }
    golden_prefill_indexer_compressor(compressor_tensors)
    tensors["idx_kv_cache"][:] = compressor_tensors["idx_kv_cache"]
    tensors["idx_kv_scale"][:] = compressor_tensors["idx_kv_scale"]

    # --- Real lightning-indexer score + per-token causal-masked top-k ---
    # Ports the official model.py Indexer.forward(start_pos==0) branch (and the deleted #505^
    # prefill indexer): score each token's query against the compressed index KV through the
    # W8A8C16 int8 path, causal-mask each token to the positions it can reach ((pos+1)//ratio),
    # then top-k. Replaces the old placeholder (sequential arange+offset, i.e. the dense
    # get_compress_topk_idxs pattern, which never exercised real selection).
    num_tokens = int(tensors["num_tokens"])
    position_ids = tensors["position_ids"].long()
    rd = ROPE_HEAD_DIM
    cmp_topk_indices = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
    score_full = torch.full((T, INDEXER_SCORE_CAP), FP32_NEG_INF, dtype=torch.float32)
    visible = ((position_ids + 1) // COMPRESS_RATIO).clamp(max=INDEXER_SCORE_CAP)
    max_visible = int(visible[:num_tokens].max().item()) if num_tokens > 0 else 0
    if max_visible == 0:
        return cmp_topk_indices, score_full

    # Q: int8 qr x int8 wq_b -> dequant -> per-token interleaved RoPE -> Hadamard rotation.
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    hadamard = tensors["hadamard"].float()
    cos = tensors["cos"].float().view(T, 1, -1)
    sin = tensors["sin"].float().view(T, 1, -1)
    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(T, IDX_N_HEADS, IDX_HEAD_DIM)
    q_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    q0, q1 = q_pair[..., 0], q_pair[..., 1]
    y0 = (q0 * cos - q1 * sin).to(torch.bfloat16)
    y1 = (q0 * sin + q1 * cos).to(torch.bfloat16)
    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
    q = q.to(torch.bfloat16).float() @ hadamard

    weights = (tensors["x"].float() @ tensors["weights_proj"].float()) * WEIGHTS_SCALE  # [T, heads]

    # C8: the compressor already stored INT8 KV + a per-position dequant scale. Gather both in
    # compressed-position order through the paged block table (no score-time re-quant).
    cache_flat_i8 = tensors["idx_kv_cache"].reshape(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
    scale_flat = tensors["idx_kv_scale"].float().reshape(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, 1)
    idx_block_table = tensors["idx_block_table"]
    rows = [
        int(idx_block_table[c // BLOCK_SIZE].item()) * BLOCK_SIZE + (c % BLOCK_SIZE)
        for c in range(max_visible)
    ]
    kv_i8 = torch.stack([cache_flat_i8[r] for r in rows], dim=0).to(torch.int32)  # [max_visible, dim]
    kv_sc = torch.stack([scale_flat[r] for r in rows], dim=0).view(1, 1, max_visible)

    # W8A8C16 int8 score, matching decode_indexer: per-row quantize q, INT32 matmul against the
    # pre-quantized KV, then dequantize by both scales before the FP32 head-weighted reduce.
    q_i8, q_sc = _int8_quant_per_row(q.reshape(T * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(T, IDX_N_HEADS, IDX_HEAD_DIM).to(torch.int32)
    q_sc = q_sc.view(T, IDX_N_HEADS, 1)
    score_i32 = torch.einsum("thd,cd->thc", q_i8, kv_i8)
    score = score_i32.float() * q_sc * kv_sc
    score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)  # [T, max_visible]

    # Per-token causal mask, then top-k over the visible compressed positions (+ raw-index offset).
    col = torch.arange(max_visible).unsqueeze(0)
    score = score.masked_fill(col >= visible.unsqueeze(1), FP32_NEG_INF)
    score_full[:, :max_visible] = score
    for t in range(num_tokens):
        k = int(min(INDEXER_TOPK_CAP, int(visible[t].item())))
        if k > 0:
            sel = score[t].topk(k, dim=-1)[1]
            cmp_topk_indices[t, :k] = sel.to(torch.int32) + INDEXER_OFFSET
    return cmp_topk_indices, score_full










