from config import D, TOPK, VOCAB

import pypto.language as pl

from config import CASE_CONFIG as M, MOE_TOKENS, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS

T = MOE_TOKENS

NORM_EPS = M.rms_norm_eps

N_EXPERTS = M.n_routed_experts

ROUTE_SCALE = M.routed_scaling_factor

N_HASH_LAYERS = M.num_hash_layers

T_TILE = 8

GATE_T_TILE = 8

GATE_M_TILE = 16        # cube M-tile: matmul rows must be a multiple of 16 (fractal)

GATE_N_TILE = 16        # expert columns per gate spmd block

T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE

D_TILE = 256

GATE_D_TILE = 256

QUANT_TILE = 256

SCORE_PAD = 256         # padded expert row for sort32 + mrgsort

TOPK_PAD = 8            # TOPK padded to 32B-aligned width

SORT_PAD = TOPK_PAD * 2 # (val, idx) interleaved slice width

@pl.jit.inline
def gate(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    x_norm: pl.Tensor[[T, D], pl.BF16],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
):
    x_norm_gate_buf = pl.create_tensor([T_PAD, D], dtype=pl.FP32)
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

    for t0 in pl.parallel(0, active_gate_tokens, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_norm"):
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for rms_d0 in pl.pipeline(0, D, D_TILE, stage=2):
                rms_x = pl.cast(x_mixed[t0 : t0 + T_TILE, rms_d0 : rms_d0 + D_TILE], pl.FP32)
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x, rms_x)), [1, T_TILE]))
            inv_rms = pl.reshape(pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS))), [T_TILE, 1])
            for an_d0 in pl.pipeline(0, D, D_TILE, stage=2):
                an_x = pl.cast(x_mixed[t0 : t0 + T_TILE, an_d0 : an_d0 + D_TILE], pl.FP32)
                an_w = pl.cast(pl.reshape(norm_w[an_d0 : an_d0 + D_TILE], [1, D_TILE]), pl.FP32)
                an_normed = pl.col_expand_mul(pl.row_expand_mul(an_x, inv_rms), an_w)
                an_bf16 = pl.cast(an_normed, pl.BF16, mode="rint")
                x_norm_gate_buf[t0 : t0 + T_TILE, an_d0 : an_d0 + D_TILE] = pl.cast(an_bf16, pl.FP32)
                x_norm[t0 : t0 + T_TILE, an_d0 : an_d0 + D_TILE] = an_bf16

    for t0 in pl.parallel(0, active_gate_tokens, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_norm_quant"):
            xn_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for xq_a_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
                xn_a_f32 = x_norm_gate_buf[t0 : t0 + T_TILE, xq_a_k : xq_a_k + QUANT_TILE]
                xn_a_abs = pl.maximum(xn_a_f32, pl.neg(xn_a_f32))
                xn_a_max = pl.reshape(pl.row_max(xn_a_abs), [1, T_TILE])
                xn_amax = pl.maximum(xn_amax, xn_a_max)
            xn_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), xn_amax)
            x_norm_scale[t0 : t0 + T_TILE, 0:1] = pl.reshape(pl.recip(xn_sq_row), [T_TILE, 1])
            xn_sq_col = pl.reshape(xn_sq_row, [T_TILE, 1])
            for xq_b_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
                xn_q_scaled = pl.row_expand_mul(
                    x_norm_gate_buf[t0 : t0 + T_TILE, xq_b_k : xq_b_k + QUANT_TILE],
                    xn_sq_col,
                )
                xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
                xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
                x_norm_i8[t0 : t0 + T_TILE, xq_b_k : xq_b_k + QUANT_TILE] = \
                    pl.cast(xn_q_half, pl.INT8, mode="trunc")

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
        for zt in pl.range(T):
            if zt >= active_tokens:
                pl.write(x_norm_scale, [zt, 0], pl.cast(0.0, pl.FP32))
                for zk in pl.range(TOPK):
                    pl.write(indices, [zt, zk], pl.cast(0, pl.INT32))
                    pl.write(weights, [zt, zk], pl.cast(0.0, pl.FP32))
        if N_EXPERTS < SCORE_PAD:
            biased_scores_buf[:, N_EXPERTS:SCORE_PAD] = \
                pl.full([T_PAD, SCORE_PAD - N_EXPERTS], dtype=pl.FP32, value=FP32_NEG_INF)

    GATE_N_BLOCKS = N_EXPERTS // GATE_N_TILE
    for gb_idx in pl.spmd(active_gate_tiles * GATE_N_BLOCKS, name_hint="gate"):
        tg = gb_idx // GATE_N_BLOCKS
        nb = gb_idx % GATE_N_BLOCKS
        t1 = tg * GATE_M_TILE
        n0 = nb * GATE_N_TILE
        gp_bias_row = pl.reshape(gate_bias[n0 : n0 + GATE_N_TILE], [1, GATE_N_TILE])
        gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
        for kb in pl.range(0, D // GATE_D_TILE):
            gd_kd = kb * GATE_D_TILE
            gd_x = x_norm_gate_buf[t1 : t1 + GATE_M_TILE, gd_kd : gd_kd + GATE_D_TILE]
            gd_w = gate_w[n0 : n0 + GATE_N_TILE, gd_kd : gd_kd + GATE_D_TILE]
            if gd_kd == 0:
                gate_logits_tile = pl.matmul(gd_x, gd_w, out_dtype=pl.FP32, b_trans=True)
            else:
                gate_logits_tile = pl.matmul_acc(gate_logits_tile, gd_x, gd_w, b_trans=True)
        gp_relu = pl.maximum(gate_logits_tile, 0.0)
        gp_abs = pl.maximum(gate_logits_tile, pl.neg(gate_logits_tile))
        gp_softplus_log = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
        gp_neg_floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(gate_logits_tile), 10.0), 0.0), 1.0)
        gp_neg_floor = pl.mul(gp_neg_floor_mask, pl.exp(pl.minimum(gate_logits_tile, 0.0)))
        gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
        gp_score = pl.sqrt(gp_softplus)
        route_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_score
        if layer_id >= N_HASH_LAYERS:
            gp_bias = pl.col_expand_mul(pl.full([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32, value=1.0), gp_bias_row)
            gp_biased = pl.add(gp_score, gp_bias)
            biased_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_biased

    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    if layer_id < N_HASH_LAYERS:
        for th_idx in pl.spmd(active_route_tiles, name_hint="route_hash"):
            t1 = th_idx * GATE_T_TILE
            hs_vals_buf = pl.full([GATE_T_TILE, TOPK_PAD], dtype=pl.FP32, value=0.0)
            hs_idx_buf = pl.full([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32, value=0)
            for hs_tt in pl.range(GATE_T_TILE):
                hs_token = pl.cast(pl.read(input_ids, [t1 + hs_tt]), pl.INDEX)
                for hs_k in pl.range(TOPK):
                    hs_eid = pl.read(tid2eid, [hs_token, hs_k])
                    hs_epos = pl.cast(hs_eid, pl.INDEX)
                    hs_unbiased = pl.read(route_scores_buf, [t1 + hs_tt, hs_epos])
                    pl.write(hs_idx_buf, [hs_tt, hs_k], hs_eid)
                    pl.write(hs_vals_buf, [hs_tt, hs_k], hs_unbiased)
            hs_denom = pl.reshape(pl.row_sum(hs_vals_buf), [GATE_T_TILE, 1])
            hs_weights_buf = pl.mul(pl.row_expand_div(hs_vals_buf, hs_denom), ROUTE_SCALE)
            for hs_wt_tt in pl.range(GATE_T_TILE):
                if t1 + hs_wt_tt < active_tokens:
                    for hs_wt_k in pl.range(TOPK):
                        pl.write(indices, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_idx_buf, [hs_wt_tt, hs_wt_k]))
                        pl.write(weights, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_weights_buf, [hs_wt_tt, hs_wt_k]))
    else:
        for ts_idx in pl.spmd(active_route_tiles, name_hint="route_sort"):
            t1 = ts_idx * GATE_T_TILE
            topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            for sr_tt in pl.range(GATE_T_TILE):
                sr_row = biased_scores_buf[t1 + sr_tt : t1 + sr_tt + 1, :]
                sr_idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                sr_sorted = pl.sort32(sr_row, sr_idx_init)
                sr_sorted = pl.mrgsort(sr_sorted, block_len=64)
                sr_sorted = pl.mrgsort(sr_sorted[:, 0:256], sr_sorted[:, 256:512])
                sr_pairs = sr_sorted[:, 0:SORT_PAD]
                sr_i = pl.gather(sr_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i
            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + GATE_T_TILE, :]
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
