import pypto.language as pl
from config import CASE_CONFIG as M, BLOCK_SIZE, PREFILL_BATCH, PREFILL_CMP_BLOCK_NUM, PREFILL_CMP_MAX_BLOCKS, PREFILL_IDX_BLOCK_NUM, PREFILL_ORI_BLOCK_NUM, PREFILL_ORI_MAX_BLOCKS, PREFILL_SEQ
from prefill_compressor_ratio4 import CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_STATE_MAX_BLOCKS, prefill_compressor_ratio4
from hc_post import hc_post_prefill
from hc_pre import hc_pre
from prefill_indexer import IDX_CACHE_MAX_BLOCKS, INDEXER_SCORE_CAP, prefill_indexer
from prefill_indexer_compressor import INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_MAX_BLOCKS
from qkv_proj_rope import materialize_rope_rows, qkv_proj_rope
from rmsnorm import rms_norm
from prefill_sparse_attn import prefill_sparse_attn
from config import B
from config import S
from config import T
from config import D
from config import H
from config import HEAD_DIM
from config import ROPE_HEAD_DIM
HALF_ROPE = ROPE_HEAD_DIM // 2
from config import Q_LORA
from config import MAX_SEQ_LEN
WIN = M.sliding_window
COMPRESS_RATIO = 4
from config import IDX_HEAD_DIM
from config import IDX_N_HEADS
IDX_TOPK = M.index_topk
from config import HC_MULT
from config import MIX_HC
from config import HC_DIM
from config import O_LORA
from config import O_GROUPS
from config import O_GROUP_IN
COFF = 2
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
from config import INNER_OUT_DIM
INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
CSA_TOPK_TOKEN_TILE = 2
from config import SPARSE_ORI_MAX_BLOCKS
from config import SPARSE_CMP_MAX_BLOCKS
from config import CSA_ORI_BLOCK_NUM
from config import CSA_CMP_BLOCK_NUM
ORI_BLOCK_NUM_DYN = pl.dynamic('PREFILL_ORI_BLOCK_NUM_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('PREFILL_CMP_BLOCK_NUM_DYN')
IDX_BLOCK_NUM_DYN = pl.dynamic('PREFILL_IDX_BLOCK_NUM_DYN')
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic('PREFILL_CSA_STATE_BLOCK_NUM_DYN')
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic('PREFILL_INNER_STATE_BLOCK_NUM_DYN')

@pl.jit.inline
def prefill_attention_csa(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32], compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32], inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32], ori_slot_mapping: pl.Tensor[[T], pl.INT64], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32], idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32], position_ids: pl.Tensor[[T], pl.INT32], cmp_slot_mapping: pl.Tensor[[T], pl.INT64], idx_slot_mapping: pl.Tensor[[T], pl.INT64], state_slot_mapping: pl.Tensor[[T], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T], pl.INT64], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]], num_tokens: pl.Scalar[pl.INT32]):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)
    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(freqs_cos, freqs_sin, position_ids, num_tokens, rope_cos_t, rope_sin_t)
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(x_normed, wq_a, wq_b, wq_b_scale, wkv, rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv, q, kv, qr, qr_scale, late_dep)
    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_csa_cache_write'):
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row:write_row + 1, :] = kv[write_t:write_t + 1, :]
    compressor_completion = pl.array.create(1, pl.TASK_ID)
    prefill_compressor_ratio4(x_normed, compress_state, compress_state_block_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, freqs_cos, freqs_sin, cmp_kv, position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping, compressor_completion)
    idx_cos = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    idx_sin = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_csa_idx_halfrope', deps=[compressor_completion[0]]):
        for idx_t in pl.range(T):
            idx_pos = pl.cast(pl.read(position_ids, [idx_t]), pl.INDEX)
            idx_cos = pl.assemble(idx_cos, pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [idx_pos, 0]), target_type=pl.FP32), [idx_t, 0])
            idx_sin = pl.assemble(idx_sin, pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [idx_pos, 0]), target_type=pl.FP32), [idx_t, 0])
    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    idx_score_unused = pl.create_tensor([T, INDEXER_SCORE_CAP], dtype=pl.FP32)
    prefill_indexer(x_normed, qr, qr_scale, idx_wq_b, idx_wq_b_scale, idx_weights_proj, idx_cos, idx_sin, freqs_cos, freqs_sin, hadamard_idx, inner_compress_state, inner_compress_state_block_table, inner_wkv, inner_wgate, inner_ape, inner_norm_w, idx_kv_cache, idx_kv_scale, idx_block_table, idx_score_unused, cmp_topk_indices, position_ids, num_tokens, idx_slot_mapping, inner_state_slot_mapping)
    swa_indices = pl.create_tensor([T, WIN], dtype=pl.INT32)
    cmp_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    for topk_block in pl.spmd((T + CSA_TOPK_TOKEN_TILE - 1) // CSA_TOPK_TOKEN_TILE, name_hint='prefill_csa_sparse_idx_tile'):
        topk_t0 = topk_block * CSA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(CSA_TOPK_TOKEN_TILE):
            t_idx = topk_t0 + topk_dt
            swa_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            cmp_row = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            if t_idx < num_tokens:
                abs_pos = pl.read(position_ids, [t_idx])
                window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for win_col in pl.range(WIN):
                    win_col_i32 = pl.cast(win_col, pl.INT32)
                    if win_col_i32 < window_valid:
                        key_abs = key_start_abs + win_col_i32
                        blk_slot = key_abs // BLOCK_SIZE
                        blk = pl.read(ori_block_table, [pl.cast(blk_slot, pl.INDEX)])
                        if blk >= 0:
                            row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                            pl.write(swa_row, [0, win_col], row)
                visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
                for cmp_col in pl.range(IDX_TOPK):
                    cmp_col_i32 = pl.cast(cmp_col, pl.INT32)
                    if cmp_col_i32 < visible_cmp:
                        topk_raw = pl.read(cmp_topk_indices, [t_idx, cmp_col])
                        if topk_raw >= 0:
                            if topk_raw < pl.cast(SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE, pl.INT32):
                                pl.write(cmp_row, [0, cmp_col], topk_raw)
            swa_indices = pl.assemble(swa_indices, swa_row, [t_idx, 0])
            cmp_indices = pl.assemble(cmp_indices, cmp_row, [t_idx, 0])
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    prefill_sparse_attn(q, kv_cache, swa_indices, cmp_kv, cmp_block_table, cmp_indices, attn_sink, num_tokens, rope_cos_t, rope_sin_t, wo_a, wo_b, wo_b_scale, attn_out)
    hc_post_prefill(attn_out, x_hc, post, comb, x_out, num_tokens)
    return (kv_cache, cmp_kv, compress_state, idx_kv_cache, idx_kv_scale, inner_compress_state, x_out)
