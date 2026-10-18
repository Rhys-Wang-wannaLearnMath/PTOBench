"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_swa_flash."""

from config import CASE_CONFIG as M
from config import TP_SIZE
from config import BLOCK_SIZE
import pypto.language as pl
import pypto.language.distributed as pld
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from decode_o_proj import LOCAL_T_PAD
from decode_o_proj import LOCAL_O_GROUPS
from decode_o_proj import o_group_a2a
from decode_o_proj import ATTENTION_WINDOW_ROWS
from decode_o_proj import O_WINDOW_ROWS
from decode_o_proj import decode_sharded_o_projection_reduce_scatter
from decode_o_proj import GROUP_T_PAD
from decode_o_proj import LOCAL_O_WIDTH
from decode_sparse_attn_swa import T_PAD
from decode_sparse_attn_swa import PADDED_TOPK
from decode_sparse_attn_swa import sparse_attn_swa
from decode_sparse_attn_swa import ATTN_K_TILE

T_DYN = pl.dynamic('T_DYN')
ORI_BLOCK_NUM_DYN = pl.dynamic('ORI_BLOCK_NUM_DYN')
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
BIAS_T_TILE = 8
NEG_INF = -1e+20

@pl.jit.inline
def decode_swa(x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], swa_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], swa_lens: pl.Tensor[[T_DYN], pl.INT32], position_ids: pl.Tensor[[T_DYN], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16], attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32], o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32]):
    """Run the complete SWA layer with tensor-parallel output."""
    t_dim = pl.tensor.dim(x_hc, 0)
    bias_blocks = t_dim // BIAS_T_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)
    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(x_normed_t, wq_a, wq_b, wq_b_scale, wkv, freqs_cos, freqs_sin, gamma_cq, gamma_ckv, q, kv, qr, qr_scale, late_dep)
    ori_block_num = pl.tensor.dim(kv_cache, 0)
    cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [cache_rows, HEAD_DIM])
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='swa_cache_insert_valid_bias'):
        for write_t in pl.range(t_dim):
            write_row_i64 = pl.read(swa_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row:write_row + 1, 0:HEAD_DIM] = kv[write_t:write_t + 1, 0:HEAD_DIM]
        valid_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        for bias_block in pl.range(bias_blocks):
            token_start = bias_block * BIAS_T_TILE
            zero_rows = pl.full([BIAS_T_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            valid_cols = pl.col_expand(zero_rows, valid_col)
            lens_slice = swa_lens[token_start:token_start + BIAS_T_TILE]
            lens_col = pl.reshape(lens_slice, [BIAS_T_TILE, 1])
            lens_fp32 = pl.cast(lens_col, target_type=pl.FP32)
            valid = pl.neg(pl.row_expand_sub(valid_cols, lens_fp32))
            valid = pl.maximum(valid, 0.0)
            valid = pl.minimum(valid, 1.0)
            invalid = pl.sub(valid, 1.0)
            sparse_bias[token_start:token_start + BIAS_T_TILE, 0:ATTN_K_TILE] = pl.mul(invalid, -NEG_INF)
    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    sparse_attn_swa(q, kv_cache, swa_indices, sparse_bias, attn_sink, freqs_cos, freqs_sin, o_packed_heads)
    attention_grouped = pl.reshape(o_packed_heads, [O_GROUPS * LOCAL_T_PAD, O_GROUP_IN])
    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    (attention_local_flat, attention_signal) = o_group_a2a(attention_grouped, attention_local_flat, attention_window, attention_signal, group_base, tp_rank, local_t)
    attention_local_groups = pl.reshape(attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN])
    o_local = pl.create_tensor([LOCAL_T_PAD, D], dtype=pl.BF16)
    (o_local, o_signal) = decode_sharded_o_projection_reduce_scatter(attention_local_groups, wo_a, wo_b, wo_b_scale, local_t, o_local, o_window, o_signal, group_base, tp_rank)
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='swa_o_local'):
        for token_start in pl.range(0, t_dim, BIAS_T_TILE):
            o_local_rows = o_local[token_start:token_start + BIAS_T_TILE, 0:D]
            attn_out[token_start:token_start + BIAS_T_TILE, 0:D] = o_local_rows
    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out
