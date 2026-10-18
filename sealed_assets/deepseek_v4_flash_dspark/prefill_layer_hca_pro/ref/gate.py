"""REF closure for deepseek_v4_flash_dspark.prefill_layer_hca_pro; fixed bindings from public config."""
from config import D
from config import FP32_NEG_INF
from config import INT8_AMAX_EPS
from config import INT8_SCALE_MAX
from config import NORM_EPS
from config import N_EXPERTS_GLOBAL as N_EXPERTS
from config import N_HASH_LAYERS
from config import ROUTE_SCALE
from config import T
from config import TOPK
from config import VOCAB
import pypto.language as pl
T_TILE = 8
GATE_T_TILE = 8
GATE_M_TILE = 16
GATE_N_TILE = 16
T_PAD = (T + GATE_M_TILE - 1) // GATE_M_TILE * GATE_M_TILE
ROW_PAD = 8
FFN_REDUCE_TILE = D // ROW_PAD
GATE_D_TILE = 1024
QUANT_TILE = 256
SCORE_PAD = 256
TOPK_PAD = 8
SORT_PAD = TOPK_PAD * 2

@pl.jit.inline
def gate(x_mixed: pl.Tensor[[T, D], pl.BF16], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32], layer_id: pl.Scalar[pl.INT32], num_tokens: pl.Scalar[pl.INT32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[T], pl.INT64], x_norm_i8: pl.Tensor[[T, D], pl.INT8], x_norm_scale: pl.Tensor[[T, 1], pl.FP32], indices: pl.Tensor[[T, TOPK], pl.INT32], weights: pl.Tensor[[T, TOPK], pl.FP32]):
    xg_buf = pl.create_tensor([T_PAD, D], dtype=pl.FP32)
    inv_rms_buf = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    xn_scale_buf = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    route_scores_buf = pl.create_tensor([T_PAD, SCORE_PAD], dtype=pl.FP32)
    biased_scores_buf = pl.create_tensor([T_PAD, SCORE_PAD], dtype=pl.FP32)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    active_gate_tokens = active_gate_tiles * GATE_M_TILE
    if active_gate_tokens > T:
        active_gate_tokens = pl.cast(T, pl.INDEX)
    norm_w_2d = pl.reshape(norm_w, [1, D])
    for tok in pl.spmd(active_gate_tokens, name_hint='ffn_norm', allow_early_resolve=True):
        rms_x = pl.cast(pl.tile.load(x_mixed, [tok, 0], [1, D]), pl.FP32)
        rms_w = pl.cast(pl.tile.load(norm_w_2d, [0, 0], [1, D]), pl.FP32)
        xg = pl.mul(rms_x, rms_w)
        pl.tile.store(xg, [tok, 0], xg_buf, shapes=[1, D])
        sq_rows = pl.reshape(pl.mul(rms_x, rms_x), [ROW_PAD, FFN_REDUCE_TILE])
        sq_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
        sq_partial = pl.row_sum(sq_rows, sq_partial_tmp)
        sq_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        sq_reduce[0:1, :] = pl.reshape(sq_partial, [1, ROW_PAD])
        sq_reduce = pl.set_validshape(sq_reduce, 1, ROW_PAD)
        sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        sq_sum = pl.row_sum(sq_reduce, sq_sum_tmp)
        sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
        inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
        pl.tile.store(inv_rms, [tok, 0], inv_rms_buf, shapes=[1, 1])
        xg_abs_rows = pl.reshape(pl.abs(xg), [ROW_PAD, FFN_REDUCE_TILE])
        amax_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
        amax_partial = pl.row_max(xg_abs_rows, amax_partial_tmp)
        amax_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        amax_reduce[0:1, :] = pl.reshape(amax_partial, [1, ROW_PAD])
        amax_reduce = pl.set_validshape(amax_reduce, 1, ROW_PAD)
        amax_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        xg_amax = pl.row_max(amax_reduce, amax_tmp)
        xg_amax = pl.set_validshape(pl.reshape(xg_amax, [1, ROW_PAD]), 1, 1)
        amax_eps = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_AMAX_EPS)
        amax_eps = pl.set_validshape(amax_eps, 1, 1)
        xg_amax = pl.maximum(xg_amax, amax_eps)
        scale_max = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX)
        scale_max = pl.set_validshape(scale_max, 1, 1)
        xg_sq = pl.mul(pl.recip(xg_amax), INT8_SCALE_MAX)
        xg_dequant_scale = pl.mul(xg_amax, 1.0 / INT8_SCALE_MAX)
        x_norm_dequant_scale = pl.mul(xg_dequant_scale, inv_rms)
        pl.tile.store(x_norm_dequant_scale, [tok, 0], x_norm_scale, shapes=[1, 1])
        pl.tile.store(xg_sq, [tok, 0], xn_scale_buf, shapes=[1, 1])
    seed_dummy = pl.system.task_dummy(deps=[])
    for t0 in pl.parallel(0, active_gate_tokens, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='x_norm_quant', deps=[seed_dummy], allow_early_resolve=True):
            xn_sq_col = xn_scale_buf[t0:t0 + T_TILE, 0:1]
            for xq_b_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
                xn_q_scaled = pl.row_expand_mul(xg_buf[t0:t0 + T_TILE, xq_b_k:xq_b_k + QUANT_TILE], xn_sq_col)
                xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode='rint')
                xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode='round')
                x_norm_i8[t0:t0 + T_TILE, xq_b_k:xq_b_k + QUANT_TILE] = pl.cast(xn_q_half, pl.INT8, mode='trunc')
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='gate_pre_route'):
        for zt in pl.range(T):
            if zt >= active_tokens:
                pl.write(x_norm_scale, [zt, 0], pl.cast(0.0, pl.FP32))
                for zk in pl.range(TOPK):
                    pl.write(indices, [zt, zk], pl.cast(0, pl.INT32))
                    pl.write(weights, [zt, zk], pl.cast(0.0, pl.FP32))
        if N_EXPERTS < SCORE_PAD:
            biased_scores_buf[:, N_EXPERTS:SCORE_PAD] = pl.full([T_PAD, SCORE_PAD - N_EXPERTS], dtype=pl.FP32, value=FP32_NEG_INF)
    GATE_N_BLOCKS = N_EXPERTS // GATE_N_TILE
    for gb_idx in pl.spmd(active_gate_tiles * GATE_N_BLOCKS, name_hint='gate', allow_early_resolve=True):
        tg = gb_idx // GATE_N_BLOCKS
        nb = gb_idx % GATE_N_BLOCKS
        t1 = tg * GATE_M_TILE
        n0 = nb * GATE_N_TILE
        gp_bias_row = pl.reshape(gate_bias[n0:n0 + GATE_N_TILE], [1, GATE_N_TILE])
        gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // GATE_D_TILE, stage=2):
            gd_kd = kb * GATE_D_TILE
            gd_x = xg_buf[t1:t1 + GATE_M_TILE, gd_kd:gd_kd + GATE_D_TILE]
            gd_w = gate_w[n0:n0 + GATE_N_TILE, gd_kd:gd_kd + GATE_D_TILE]
            if gd_kd == 0:
                gate_logits_tile = pl.matmul(gd_x, gd_w, out_dtype=pl.FP32, b_trans=True)
            else:
                gate_logits_tile = pl.matmul_acc(gate_logits_tile, gd_x, gd_w, b_trans=True)
        gate_logits_tile = pl.row_expand_mul(gate_logits_tile, inv_rms_buf[t1:t1 + GATE_M_TILE, 0:1])
        gp_relu = pl.maximum(gate_logits_tile, 0.0)
        gp_abs = pl.maximum(gate_logits_tile, pl.neg(gate_logits_tile))
        gp_softplus_log = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
        gp_neg_floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(gate_logits_tile), 10.0), 0.0), 1.0)
        gp_neg_floor = pl.mul(gp_neg_floor_mask, pl.exp(pl.minimum(gate_logits_tile, 0.0)))
        gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
        gp_score = pl.sqrt(gp_softplus)
        route_scores_buf[t1:t1 + GATE_M_TILE, n0:n0 + GATE_N_TILE] = gp_score
        if layer_id >= N_HASH_LAYERS:
            gp_bias = pl.col_expand_mul(pl.full([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32, value=1.0), gp_bias_row)
            gp_biased = pl.add(gp_score, gp_bias)
            biased_scores_buf[t1:t1 + GATE_M_TILE, n0:n0 + GATE_N_TILE] = gp_biased
    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    if layer_id < N_HASH_LAYERS:
        for th_idx in pl.spmd(active_route_tiles, name_hint='route_hash', allow_early_resolve=True):
            t1 = th_idx * GATE_T_TILE
            hs_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            for hs_tt in pl.range(GATE_T_TILE):
                hs_token = pl.cast(pl.read(input_ids, [t1 + hs_tt]), pl.INDEX)
                for hs_k in pl.range(TOPK):
                    pl.write(hs_idx_tile, [hs_tt, hs_k], pl.read(tid2eid, [hs_token, hs_k]))
                for hs_pad_k in pl.range(TOPK, TOPK_PAD):
                    pl.write(hs_idx_tile, [hs_tt, hs_pad_k], pl.cast(0, pl.INT32))
            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1:t1 + GATE_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=hs_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            hs_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            hs_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            hs_idx_read[:, :] = hs_idx_tile[:, :]
            hs_denom = pl.reshape(pl.row_sum(hs_vals_pad), [GATE_T_TILE, 1])
            hs_weights_pad = pl.mul(pl.row_expand_div(hs_vals_pad, hs_denom), ROUTE_SCALE)
            for hs_wt_tt in pl.range(GATE_T_TILE):
                if t1 + hs_wt_tt < active_tokens:
                    for hs_wt_k in pl.range(TOPK):
                        pl.write(indices, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_idx_read, [hs_wt_tt, hs_wt_k]))
                        pl.write(weights, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_weights_pad, [hs_wt_tt, hs_wt_k]))
    else:
        for ts_idx in pl.spmd(active_route_tiles, name_hint='route_sort', allow_early_resolve=True):
            t1 = ts_idx * GATE_T_TILE
            topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            for sr_tt in pl.range(GATE_T_TILE):
                sr_row = biased_scores_buf[t1 + sr_tt:t1 + sr_tt + 1, :]
                sr_idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                sr_sorted = pl.sort32(sr_row, sr_idx_init)
                sr_sorted = pl.mrgsort(sr_sorted, block_len=64)
                sr_sorted = pl.mrgsort(sr_sorted[:, 0:256], sr_sorted[:, 256:512])
                sr_pairs = sr_sorted[:, 0:SORT_PAD]
                sr_i = pl.gather(sr_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                topk_idx_tile[sr_tt:sr_tt + 1, :] = sr_i
            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1:t1 + GATE_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            topk_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            topk_idx_read[:, :] = topk_idx_tile[:, :]
            nm_denom = pl.reshape(pl.row_sum(topk_vals_pad), [GATE_T_TILE, 1])
            nm_weights_pad = pl.mul(pl.row_expand_div(topk_vals_pad, nm_denom), ROUTE_SCALE)
            for nm_tt in pl.range(GATE_T_TILE):
                if t1 + nm_tt < active_tokens:
                    for nm_k in pl.range(TOPK):
                        pl.write(indices, [t1 + nm_tt, nm_k], pl.read(topk_idx_read, [nm_tt, nm_k]))
                        pl.write(weights, [t1 + nm_tt, nm_k], pl.read(nm_weights_pad, [nm_tt, nm_k]))
    return weights
