"""NPU Ref for the fixed legacy DeepSeek-V4 ratio-128 decode compressor."""

import pypto.language as pl

from config import (
    APE_ROW,
    B,
    COMPRESS_RATIO,
    D,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    NOPE_HEAD_DIM,
    OUT_DIM,
    ROPE_HEAD_DIM,
    S,
    SCATTER_SLOT,
    STATE_LEN,
)


# Implementation tiling choices remain editable after publication.
K_CHUNK = 512
OUT_CHUNK = 64
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK
HEAD_CHUNK = 128
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK


@pl.jit.inline
def deepseek_v4_decode_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Tensor[[B, HEAD_DIM], pl.BF16],
):
    x_flat = pl.reshape(x, [B, D])
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])

    kv_fp32 = pl.create_tensor([B, OUT_DIM], dtype=pl.FP32)
    score_fp32 = pl.create_tensor([B, OUT_DIM], dtype=pl.FP32)
    slot_off = SCATTER_SLOT * OUT_DIM

    for ob in pl.parallel(0, OUT_BLOCKS, 1):
        oc0 = ob * OUT_CHUNK
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj"):
            a0 = pl.slice(x_flat, [B, K_CHUNK], [0, 0])
            b0 = pl.slice(wkv, [OUT_CHUNK, K_CHUNK], [oc0, 0])
            kv_acc = pl.matmul(a0, b0, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.range(1, K_BLOCKS):
                a_i = pl.slice(x_flat, [B, K_CHUNK], [0, kb * K_CHUNK])
                b_i = pl.slice(wkv, [OUT_CHUNK, K_CHUNK], [oc0, kb * K_CHUNK])
                kv_acc = pl.matmul_acc(kv_acc, a_i, b_i, b_trans=True)
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [0, oc0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_proj"):
            a0g = pl.slice(x_flat, [B, K_CHUNK], [0, 0])
            b0g = pl.slice(wgate, [OUT_CHUNK, K_CHUNK], [oc0, 0])
            sc_acc = pl.matmul(a0g, b0g, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.range(1, K_BLOCKS):
                a_ig = pl.slice(x_flat, [B, K_CHUNK], [0, kb * K_CHUNK])
                b_ig = pl.slice(wgate, [OUT_CHUNK, K_CHUNK], [oc0, kb * K_CHUNK])
                sc_acc = pl.matmul_acc(sc_acc, a_ig, b_ig, b_trans=True)
            score_fp32 = pl.assemble(score_fp32, sc_acc, [0, oc0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ape_add"):
            sc = pl.slice(score_fp32, [B, OUT_CHUNK], [0, oc0])
            ape_row = pl.slice(ape, [1, OUT_CHUNK], [APE_ROW, oc0])
            ones_b = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=1.0)
            ape_broadcast = pl.col_expand_mul(ones_b, ape_row)
            sc = pl.add(sc, ape_broadcast)
            score_fp32 = pl.assemble(score_fp32, sc, [0, oc0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter"):
            kv_chunk = pl.slice(kv_fp32, [B, OUT_CHUNK], [0, oc0])
            kv_state_flat = pl.assemble(kv_state_flat, kv_chunk, [0, slot_off + oc0])
            sc_chunk = pl.slice(score_fp32, [B, OUT_CHUNK], [0, oc0])
            score_state_flat = pl.assemble(score_state_flat, sc_chunk, [0, slot_off + oc0])

    kv_state_per_row = pl.reshape(kv_state_flat, [B * STATE_LEN, OUT_DIM])
    score_state_per_row = pl.reshape(score_state_flat, [B * STATE_LEN, OUT_DIM])

    pooled = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    for b_idx in pl.parallel(0, B, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool_loop"):
            row_b = b_idx * STATE_LEN
            for hb in pl.range(HEAD_BLOCKS):
                h0 = hb * HEAD_CHUNK
                s_max = pl.slice(score_state_per_row, [1, HEAD_CHUNK], [row_b, h0])
                for s in pl.range(1, STATE_LEN):
                    sr_max = pl.slice(score_state_per_row, [1, HEAD_CHUNK], [row_b + s, h0])
                    s_max = pl.maximum(s_max, sr_max)

                e_sum = pl.full([1, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
                weighted = pl.full([1, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
                for s in pl.range(STATE_LEN):
                    sr_exp = pl.slice(score_state_per_row, [1, HEAD_CHUNK], [row_b + s, h0])
                    kv_row = pl.slice(kv_state_per_row, [1, HEAD_CHUNK], [row_b + s, h0])
                    e_row = pl.exp(pl.sub(sr_exp, s_max))
                    e_sum = pl.add(e_sum, e_row)
                    weighted = pl.add(weighted, pl.mul(e_row, kv_row))

                pooled_chunk = pl.div(weighted, e_sum)
                pooled = pl.assemble(pooled, pooled_chunk, [b_idx, h0])

    kv_state = pl.reshape(kv_state_per_row, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_per_row, [B, STATE_LEN, OUT_DIM])

    normed_pooled = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
        partial_sq = pl.full([1, B], dtype=pl.FP32, value=0.0)
        for hb in pl.range(HEAD_BLOCKS):
            h0 = hb * HEAD_CHUNK
            pc = pl.slice(pooled, [B, HEAD_CHUNK], [0, h0])
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(pc, pc)), [1, B]),
            )
        inv_rms = pl.reshape(
            pl.recip(pl.sqrt(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS))),
            [B, 1],
        )
        for hb in pl.range(HEAD_BLOCKS):
            h0 = hb * HEAD_CHUNK
            nc = pl.slice(pooled, [B, HEAD_CHUNK], [0, h0])
            nw_chunk = pl.cast(
                pl.slice(norm_w_2d, [1, HEAD_CHUNK], [0, h0]),
                target_type=pl.FP32,
            )
            normed = pl.col_expand_mul(pl.row_expand_mul(nc, inv_rms), nw_chunk)
            normed_pooled = pl.assemble(normed_pooled, normed, [0, h0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_nope"):
        nope_chunk = pl.slice(normed_pooled, [B, NOPE_HEAD_DIM], [0, 0])
        out = pl.assemble(out, pl.cast(nope_chunk, target_type=pl.BF16), [0, 0])

    half_rd = ROPE_HEAD_DIM // 2
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_store"):
        x_lo = pl.slice(normed_pooled, [B, half_rd], [0, NOPE_HEAD_DIM])
        x_hi = pl.slice(normed_pooled, [B, half_rd], [0, NOPE_HEAD_DIM + half_rd])
        cos_fp32 = pl.cast(cos, target_type=pl.FP32)
        sin_fp32 = pl.cast(sin, target_type=pl.FP32)
        y_lo = pl.sub(pl.col_expand_mul(x_lo, cos_fp32), pl.col_expand_mul(x_hi, sin_fp32))
        y_hi = pl.add(pl.col_expand_mul(x_lo, sin_fp32), pl.col_expand_mul(x_hi, cos_fp32))
        out = pl.assemble(out, pl.cast(y_lo, target_type=pl.BF16), [0, NOPE_HEAD_DIM])
        out = pl.assemble(out, pl.cast(y_hi, target_type=pl.BF16), [0, NOPE_HEAD_DIM + half_rd])

    return out


@pl.jit
def deepseek_v4_decode_compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv_state: pl.InOut[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    score_state: pl.InOut[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[B, HEAD_DIM], pl.BF16]],
):
    out = deepseek_v4_decode_compressor(
        x, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin,
        hadamard, start_pos, out,
    )
    return out
