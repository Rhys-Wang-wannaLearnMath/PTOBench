import pypto.language as pl
from config import CASE_CONFIG as M, BLOCK_SIZE, PREFILL_BATCH, PREFILL_CMP_MAX_BLOCKS, PREFILL_ORI_MAX_BLOCKS, PREFILL_SEQ
from prefill_compressor_ratio4 import CSA_STATE_BLOCK_SIZE, CSA_STATE_MAX_BLOCKS, compressor_ratio4
from hc_post import hc_post_prefill
from hc_pre import hc_pre
from prefill_indexer import IDX_CACHE_MAX_BLOCKS, INDEXER_SCORE_CAP, INDEXER_TOPK_CAP, prefill_indexer
from prefill_indexer_compressor import INNER_STATE_BLOCK_SIZE, INNER_STATE_MAX_BLOCKS
from qkv_proj_rope import materialize_rope_rows, qkv_proj_rope
from rmsnorm import rms_norm
from prefill_sparse_attn import PREFILL_ATTN_BLOCKS, PREFILL_ATTN_TILE, SPARSE_CMP_BIAS_COLS, VALID_BLOCK_MASK_COLS, sparse_attn
from config import B, S, T, D, H, HEAD_DIM, ROPE_HEAD_DIM, HALF_ROPE, Q_LORA, MAX_SEQ_LEN, WIN, COMPRESS_RATIO, IDX_HEAD_DIM, IDX_N_HEADS, IDX_TOPK, HC_MULT, MIX_HC, HC_DIM, O_LORA, O_GROUPS, O_GROUP_IN, COFF, MAIN_OUT_DIM, MAIN_COMPRESS_STATE_DIM, INNER_OUT_DIM, INNER_COMPRESS_STATE_DIM, CMP_MAX_BLOCKS, SPARSE_ORI_MAX_BLOCKS, SPARSE_CMP_MAX_BLOCKS
ORI_BLOCK_NUM_DYN = pl.dynamic('PREFILL_ORI_BLOCK_NUM_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('PREFILL_CMP_BLOCK_NUM_DYN')
IDX_BLOCK_NUM_DYN = pl.dynamic('PREFILL_IDX_BLOCK_NUM_DYN')
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic('PREFILL_CSA_STATE_BLOCK_NUM_DYN')
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic('PREFILL_INNER_STATE_BLOCK_NUM_DYN')
CSA_TOPK_TOKEN_TILE = 2
assert S == WIN, 'packed CSA prefill currently assumes one static window page'
assert PREFILL_ATTN_BLOCKS <= VALID_BLOCK_MASK_COLS
assert INDEXER_TOPK_CAP <= SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE

@pl.jit.inline
def prefill_attention_csa(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32], compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32], inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32], ori_slot_mapping: pl.Tensor[[T], pl.INT64], cmp_kv: pl.Out[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32], position_ids: pl.Tensor[[T], pl.INT32], cmp_slot_mapping: pl.Tensor[[T], pl.INT64], idx_slot_mapping: pl.Tensor[[T], pl.INT64], state_slot_mapping: pl.Tensor[[T], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T], pl.INT64], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]], num_tokens: pl.Scalar[pl.INT32]):
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
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [ori_cache_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_csa_cache_write'):
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row:write_row + 1, :] = kv[write_t:write_t + 1, :]
    compressor_completion = pl.array.create(1, pl.TASK_ID)
    compressor_ratio4(x_normed, compress_state, compress_state_block_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, freqs_cos, freqs_sin, cmp_kv, position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping, compressor_completion)
    idx_cos = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    idx_sin = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_csa_idx_halfrope'):
        for idx_t in pl.range(T):
            idx_pos = pl.cast(pl.read(position_ids, [idx_t]), pl.INDEX)
            idx_cos = pl.assemble(idx_cos, pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [idx_pos, 0]), target_type=pl.FP32), [idx_t, 0])
            idx_sin = pl.assemble(idx_sin, pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [idx_pos, 0]), target_type=pl.FP32), [idx_t, 0])
    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    idx_score_unused = pl.create_tensor([T, INDEXER_SCORE_CAP], dtype=pl.FP32)
    (idx_kv_cache_out, idx_kv_scale_out, idx_score_unused, cmp_topk_indices) = prefill_indexer(x_normed, qr, qr_scale, idx_wq_b, idx_wq_b_scale, idx_weights_proj, idx_cos, idx_sin, freqs_cos, freqs_sin, hadamard_idx, inner_compress_state, inner_compress_state_block_table, inner_wkv, inner_wgate, inner_ape, inner_norm_w, idx_kv_cache, idx_kv_scale, idx_block_table, idx_score_unused, cmp_topk_indices, position_ids, num_tokens, idx_slot_mapping, inner_state_slot_mapping)
    swa_indices = pl.create_tensor([T, WIN], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([T, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    for topk_block in pl.spmd((T + CSA_TOPK_TOKEN_TILE - 1) // CSA_TOPK_TOKEN_TILE, name_hint='prefill_csa_sparse_idx_tile'):
        topk_t0 = topk_block * CSA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(CSA_TOPK_TOKEN_TILE):
            t_idx = topk_t0 + topk_dt
            swa_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            mask_row = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
            if t_idx < num_tokens:
                abs_pos = pl.read(position_ids, [t_idx])
                visible_cmp = pl.min((abs_pos + 1) // COMPRESS_RATIO, pl.cast(INDEXER_TOPK_CAP, pl.INT32))
                for mask_sb in pl.unroll(PREFILL_ATTN_BLOCKS):
                    cmp_lo = pl.max(mask_sb * PREFILL_ATTN_TILE - WIN, pl.cast(0, pl.INT32))
                    cmp_hi = pl.min((mask_sb + 1) * PREFILL_ATTN_TILE - WIN, pl.cast(SPARSE_CMP_BIAS_COLS, pl.INT32))
                    if cmp_lo < cmp_hi:
                        if visible_cmp > cmp_lo:
                            pl.write(mask_row, [0, mask_sb], pl.cast(1, pl.INT32))
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
                            pl.write(mask_row, [0, win_col // PREFILL_ATTN_TILE], pl.cast(1, pl.INT32))
            swa_indices = pl.assemble(swa_indices, swa_row, [t_idx, 0])
            valid_block_mask = pl.assemble(valid_block_mask, mask_row, [t_idx, 0])
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn(q, kv_cache, swa_indices, cmp_kv, cmp_block_table, cmp_topk_indices, valid_block_mask, attn_sink, num_tokens, rope_cos_t, rope_sin_t, wo_a, wo_b, wo_b_scale, attn_out)
    hc_post_prefill(attn_out, x_hc, post, comb, x_out, num_tokens)
    return (kv_cache, cmp_kv, compress_state, idx_kv_cache, idx_kv_scale, inner_compress_state, x_out)

@pl.jit
def prefill_attention_csa_test(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32], ori_slot_mapping: pl.Tensor[[T], pl.INT64], cmp_kv: pl.Out[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32], position_ids: pl.Tensor[[T], pl.INT32], cmp_slot_mapping: pl.Tensor[[T], pl.INT64], idx_slot_mapping: pl.Tensor[[T], pl.INT64], state_slot_mapping: pl.Tensor[[T], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T], pl.INT64], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]], num_tokens: pl.Scalar[pl.INT32]):
    prefill_attention_csa(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, hadamard_idx, idx_wq_b, idx_wq_b_scale, idx_weights_proj, inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_compress_state, inner_compress_state_block_table, kv_cache, ori_block_table, ori_slot_mapping, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, position_ids, cmp_slot_mapping, idx_slot_mapping, state_slot_mapping, inner_state_slot_mapping, attn_sink, wo_a, wo_b, wo_b_scale, x_out, num_tokens)
    return (kv_cache, cmp_kv, compress_state, idx_kv_cache, idx_kv_scale, inner_compress_state, x_out)
