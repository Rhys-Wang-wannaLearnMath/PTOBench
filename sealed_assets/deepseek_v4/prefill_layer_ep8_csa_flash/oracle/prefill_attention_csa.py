"""Case-local CPU Oracle for Batch 08 DeepSeek V4 prefill attention."""

from __future__ import annotations

import torch

from config import (
    CASE_CONFIG as M,
    BLOCK_SIZE,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_IDX_BLOCK_NUM,
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_SEQ,
)
from prefill_compressor_ratio4 import (
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    golden_prefill_compressor_ratio4,
)
from prefill_hc_post import golden_prefill_hc_post
from prefill_hc_pre import golden_prefill_hc_pre
from prefill_indexer import (
    IDX_CACHE_MAX_BLOCKS,
    INDEXER_SCORE_CAP,
    INDEXER_TOPK_CAP,
    _int8_quant_per_row as _idx_int8_quant_per_row,
    golden_prefill_indexer_core,
)
from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
)
from prefill_qkv_proj_rope_v2 import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from prefill_sparse_attn_v2 import (
    _quant_w_per_channel,
    golden_prefill_sparse_attn,
)
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
COMPRESS_RATIO = 4
START_POS = 0
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
IDX_HEAD_DIM = M.index_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_TOPK = M.index_topk
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
SPARSE_TOPK = WIN + IDX_TOPK
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
COFF = 2
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_LEN = COFF * COMPRESS_RATIO
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
CSA_TOPK_TOKEN_TILE = 2
WRITEBACK_GUARD_TILE = 16
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = CMP_MAX_BLOCKS
PREFILL_SPARSE_TOPK = min(SPARSE_TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
SPARSE_PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)
CSA_ORI_BLOCK_NUM = SPARSE_ORI_MAX_BLOCKS
CSA_CMP_BLOCK_NUM = CMP_BLOCK_NUM
assert S == WIN, "packed CSA prefill currently assumes one static window page"

def golden_prefill_attention_csa(tensors):
    """Torch reference for token-major packed CSA with overlay compressor/indexer."""
    import torch

    num_tokens = int(tensors["num_tokens"])
    x_hc_rect = tensors["x_hc"].view(B, S, HC_MULT, D)
    x_mixed_3d = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_3d = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    comb_4d = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
    golden_prefill_hc_pre({
        "x": x_hc_rect,
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed_3d,
        "post": post_3d,
        "comb": comb_4d,
    })
    x_mixed = x_mixed_3d.reshape(T, D)
    post = post_3d.reshape(T, HC_MULT)
    comb = comb_4d.reshape(T, HC_MULT * HC_MULT)

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    rope_cos_t = torch.zeros(T, ROPE_HEAD_DIM, dtype=torch.bfloat16)
    rope_sin_t = torch.zeros(T, ROPE_HEAD_DIM, dtype=torch.bfloat16)
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()
    golden_qkv_proj_rope({
        "x": x_normed.view(T, D),
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_t,
        "rope_sin": rope_sin_t,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
    })

    golden_prefill_compressor_ratio4({
        "x": x_normed.view(T, D),
        "kv_state": tensors["cmp_kv_state"],
        "score_state": tensors["cmp_score_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "cmp_kv": tensors["cmp_kv"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "cmp_slot_mapping": tensors["cmp_slot_mapping"],
        "state_slot_mapping": tensors["state_slot_mapping"],
    })
    idx_cos = rope_cos_t[:, :HALF_ROPE].float().contiguous()
    idx_sin = rope_sin_t[:, :HALF_ROPE].float().contiguous()
    cmp_topk_indices, _idx_score = golden_prefill_indexer_core({
        "x": x_normed.view(T, D),
        "qr": qr,
        "qr_scale": qr_scale,
        "wq_b": tensors["idx_wq_b"],
        "wq_b_scale": tensors["idx_wq_b_scale"],
        "weights_proj": tensors["idx_weights_proj"],
        "cos": idx_cos,
        "sin": idx_sin,
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "hadamard": tensors["hadamard_idx"],
        "inner_kv_state": tensors["inner_kv_state"],
        "inner_score_state": tensors["inner_score_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "inner_wkv": tensors["inner_wkv"],
        "inner_wgate": tensors["inner_wgate"],
        "inner_ape": tensors["inner_ape"],
        "inner_norm_w": tensors["inner_norm_w"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "idx_block_table": tensors["idx_block_table"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    })

    def assemble_sparse_indices(cmp_topk):
        topk_idxs = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
        sparse_lens = torch.zeros(T, dtype=torch.int32)
        pos = tensors["position_ids"]
        current = {int(pos[t].item()): t for t in range(num_tokens)}
        compressed_cap = SPARSE_PREFILL_SPARSE_PAD - WIN
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                overlay_t = current.get(key_abs)
                if overlay_t is not None and overlay_t <= t:
                    topk_idxs[t, k] = WIN + overlay_t
                else:
                    topk_idxs[t, k] = key_abs % WIN
            cursor = window_valid
            visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
            for ck, raw_t in enumerate(cmp_topk[t, :compressed_cap].tolist()):
                if cursor >= SPARSE_PREFILL_SPARSE_PAD:
                    break
                raw = int(raw_t)
                if raw >= WIN + T and raw - (WIN + T) < visible_cmp:
                    topk_idxs[t, cursor] = raw
                    cursor += 1
            sparse_lens[t] = cursor
        return topk_idxs, sparse_lens

    cmp_sparse_indices, cmp_sparse_lens = assemble_sparse_indices(cmp_topk_indices)
    kv_cache_in = tensors["kv_cache"].clone()
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": kv_cache_in,
        "ori_block_table": tensors["ori_block_table"],
        "kv_overlay": kv,
        "cmp_kv": tensors["cmp_kv"],
        "cmp_block_table": tensors["cmp_block_table"],
        "cmp_sparse_indices": cmp_sparse_indices,
        "cmp_sparse_lens": cmp_sparse_lens,
        "attn_sink": tensors["attn_sink"],
        "num_tokens": tensors["num_tokens"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    kv_cache_out = kv_cache_in.clone()
    kv_cache_flat = kv_cache_out.view(CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]
    tensors["kv_cache"][:] = kv_cache_out

    y = torch.zeros(T, HC_MULT, D, dtype=torch.bfloat16)
    y_4d = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_prefill_hc_post({
        "x": attn_out.reshape(B, S, D),
        "residual": x_hc_rect,
        "post": post_3d,
        "comb": comb_4d,
        "y": y_4d,
    })
    tensors["x_out"][:] = y_4d.reshape(T, HC_MULT, D)

