# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reachable TQ boundary generation, INT4 quantization and dequantization."""

import math as _math
import pypto.language as pl
from config import (
    EPS, HEAD_DIM, HALF_DIM, KV_HIDDEN, NUM_KV_HEADS, N_LEVELS, TQ_BITS,
    TQ_MAX_ITER, TQ_TOL, TQ_INIT_LO, TQ_INIT_HI, TQ_CDF_GUARD,
)

TOK_TILE = 16
CMP_CHUNK = 32

def solve_lloyd_max(d: int, bits: int, max_iter: int = TQ_MAX_ITER, tol: float = TQ_TOL):
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d).

    Uses fully vectorized operations:
    - Analytical Gaussian CDF via torch.erf for centroid updates
    - No per-element Python loops in the hot path

    Returns:
        centroids: sorted tensor of 2^bits optimal centroids
        boundaries: sorted tensor of 2^bits - 1 boundaries
    """
    import torch

    n_levels = 2 ** bits
    sigma = 1.0 / _math.sqrt(d)

    # Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
    lo, hi = TQ_INIT_LO, TQ_INIT_HI
    centroids = torch.linspace(lo, hi, n_levels + 2)[1:-1]  # n_levels points, excluding endpoints

    for _ in range(max_iter):
        # Step 1: boundaries = midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Step 2: update centroids via analytical Gaussian conditional mean.
        # For N(0, sigma^2) in interval [a, b]:
        #   E[X | a < X < b] = sigma * (phi(a/sigma) - phi(b/sigma))
        #                       / (Phi(b/sigma) - Phi(a/sigma))
        # where phi = standard normal PDF, Phi = CDF.
        # phi(x) = exp(-x^2/2) / sqrt(2pi), Phi(x) = 0.5*(1+erf(x/sqrt(2)))
        edges = torch.zeros(n_levels + 1)
        edges[0] = lo * 3
        edges[1:-1] = boundaries
        edges[-1] = hi * 3

        # Standardize edges
        z = edges / sigma

        # phi(z) = exp(-z^2/2) / sqrt(2*pi)
        neg_half_z2 = -0.5 * z * z
        phi_z = neg_half_z2.exp() * (1.0 / _math.sqrt(2.0 * _math.pi))

        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        cdf_z = 0.5 * (1.0 + torch.erf(z * (1.0 / _math.sqrt(2.0))))

        # Conditional means: sigma * (phi(z_a) - phi(z_b)) / (Phi(z_b) - Phi(z_a))
        phi_diff = phi_z[:-1] - phi_z[1:]   # phi(z_a) - phi(z_b), shape (n_levels,)
        cdf_diff = cdf_z[1:] - cdf_z[:-1]   # Phi(z_b) - Phi(z_a), shape (n_levels,)

        # Guard against tiny intervals
        valid = cdf_diff > TQ_CDF_GUARD
        new_centroids = torch.where(
            valid,
            sigma * phi_diff / cdf_diff.clamp(min=TQ_CDF_GUARD),
            centroids,
        )

        # Check convergence
        max_shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.clone(), boundaries.clone()

_lm_centroids, _lm_boundaries = solve_lloyd_max(
    HEAD_DIM, TQ_BITS,
)
_b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7 = [
    float(x) for x in _lm_boundaries[:8]
]
_b8, _b9, _b10, _b11, _b12, _b13, _b14 = [
    float(x) for x in _lm_boundaries[8:]
]

@pl.jit.inline
def turboquant_kv_dequant_chunk(
    quant_indices: pl.Tensor[[CMP_CHUNK, HALF_DIM], pl.UINT8],
    quant_scales: pl.Tensor[[CMP_CHUNK, 1], pl.FP32],
    tq_codebook: pl.Tensor[[CMP_CHUNK, N_LEVELS], pl.FP32],
    rot_slice: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
):
    """Dequantize one CMP_CHUNK of packed-INT4 KV cache: unpack → gather → renormalize → scale → unrotate.

    Indices are nibble-packed (two 4-bit indices per UINT8) in the half/half
    layout: byte c holds (idx[c + HALF_DIM] << 4) | idx[c].  Caller must wrap
    this call in a pl.at scope.  Caller slices a CMP_CHUNK of packed UINT8
    indices + FP32 scales from the quant cache and assembles the returned BF16
    tensor into a larger buffer.

    Steps (all inlined into the caller's scope):
      1. Unpack: UINT8 → FP16 → FP32, split into lo (fmods 16) / hi (div 16, trunc)
      2. Gather: pl.gather(codebook) per half (kept separate; no concat/assemble)
      3. Renormalize + scale: rsqrt(Σlo²+Σhi² + EPS), then ×L2 scales (each half)
      4. Unrotate: lo@rot[:,0:HALF]^T + hi@rot[:,HALF:]^T (split matmul) → cast BF16
    """
    # Unpack nibbles (half/half layout): byte c -> idx[c]=lo (mod 16),
    # idx[c+HALF_DIM]=hi (floor(/16)).  Arithmetic via unified div/fmods
    # (ands/shrs are tile-only and reject GM-slice tensors).  idx/16 is exact
    # in FP (16 = 2^4), so trunc cast gives the exact floor.  The two halves are
    # kept SEPARATE through renorm/scale/unrotate — no concat/assemble of gather
    # results (codegen's tmov rejects assembling gather sub-tiles, and concat
    # deadlocks the runtime).  Linearity makes the split exact:
    #   ||dec||² = Σlo² + Σhi² ;  dec @ rot^T = lo @ rot[:,0:HALF_DIM]^T
    #                                 + hi @ rot[:,HALF_DIM:]^T.
    idx_fp32 = pl.cast(pl.cast(quant_indices, target_type=pl.FP16), target_type=pl.FP32)
    # Derive the low nibble arithmetically.  The A2/A3 lowerer rejects fmods
    # when a GM-derived tile has lost its source extent; UINT8 values are
    # exact in FP32, so truncating the quotient preserves the packed nibble.
    hi_fp32 = pl.div(idx_fp32, 16.0)
    hi_idx = pl.cast(hi_fp32, target_type=pl.INT32, mode="trunc")  # 0..15
    lo_fp32 = pl.sub(idx_fp32, pl.mul(pl.cast(hi_idx, target_type=pl.FP32), 16.0))
    lo_idx = pl.cast(lo_fp32, target_type=pl.INT32, mode="trunc")  # 0..15

    # Expand the codebook through equality masks. Tensor-level gather creates
    # a per-row TMOV destination with valid_shape 0 on native A2/A3; summing
    # masks keeps all intermediates in the regular tensor path.
    centroids_lo = pl.full([CMP_CHUNK, HALF_DIM], dtype=pl.FP32, value=0.0)
    centroids_hi = pl.full([CMP_CHUNK, HALF_DIM], dtype=pl.FP32, value=0.0)
    for _level in pl.range(N_LEVELS):
        codebook_value = pl.read(tq_codebook, [0, _level])
        level_i32 = pl.cast(_level, target_type=pl.INT32)
        lo_mask = pl.cast(pl.cmp(lo_idx, level_i32, cmp_type=0), target_type=pl.FP32)
        hi_mask = pl.cast(pl.cmp(hi_idx, level_i32, cmp_type=0), target_type=pl.FP32)
        centroids_lo = pl.add(centroids_lo, pl.mul(lo_mask, codebook_value))
        centroids_hi = pl.add(centroids_hi, pl.mul(hi_mask, codebook_value))

    # Renormalize to unit sphere (||dec||² = Σlo² + Σhi²) + rescale by L2 norms.
    dec_sq = pl.reshape(
        pl.add(
            pl.row_sum(pl.mul(centroids_lo, centroids_lo)),
            pl.row_sum(pl.mul(centroids_hi, centroids_hi)),
        ),
        [CMP_CHUNK, 1],
    )
    inv_norm = pl.rsqrt(pl.add(dec_sq, EPS))
    lo_scaled = pl.row_expand_mul(
        pl.row_expand_mul(centroids_lo, inv_norm), quant_scales,
    )
    hi_scaled = pl.row_expand_mul(
        pl.row_expand_mul(centroids_hi, inv_norm), quant_scales,
    )
    # Unrotate: dec @ rot^T = lo @ rot_lo^T + hi @ rot_hi^T, output BF16.
    rot_lo = pl.slice(rot_slice, [HEAD_DIM, HALF_DIM], [0, 0])
    rot_hi = pl.slice(rot_slice, [HEAD_DIM, HALF_DIM], [0, HALF_DIM])
    dec_unrot = pl.add(
        pl.matmul(pl.cast(lo_scaled, target_type=pl.BF16), rot_lo,
                  b_trans=True, out_dtype=pl.FP32),
        pl.matmul(pl.cast(hi_scaled, target_type=pl.BF16), rot_hi,
                  b_trans=True, out_dtype=pl.FP32),
    )
    return pl.slice(
        pl.cast(dec_unrot, target_type=pl.BF16),
        [CMP_CHUNK, HEAD_DIM], [0, 0],
        valid_shape=[CMP_CHUNK, HEAD_DIM],
    )

@pl.jit.inline
def turboquant_kv_quantize(
    # Inputs (produced by Scope 1 in caller).
    k_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    v_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    rot_matrix: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    cos_lo_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    cos_hi_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    sin_lo_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    sin_hi_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    # Outputs (caller creates, callee fills via pl.assemble). Index temps are
    # nibble-packed: [TOK_TILE, KV_HIDDEN // 2] (two 4-bit indices per UINT8).
    quant_k_temp: pl.Tensor[[TOK_TILE, KV_HIDDEN // 2], pl.UINT8],
    quant_v_temp: pl.Tensor[[TOK_TILE, KV_HIDDEN // 2], pl.UINT8],
    k_scales_buf: pl.Tensor[[TOK_TILE, NUM_KV_HEADS], pl.FP32],
    v_scales_buf: pl.Tensor[[TOK_TILE, NUM_KV_HEADS], pl.FP32],
):
    """Batched K/V RoPE + inline quantization for prefill.

    For each KV head:
      Scope A: K RoPE + L2 norm + normalize  -> k_norm_buf, k_scales_buf
      Scope B: K rotate + Lloyd-Max quantize  -> quant_k_temp, k_rot_buf
      Scope C: V L2 norm + normalize          -> v_norm_buf, v_scales_buf
      Scope D: V rotate + Lloyd-Max quantize  -> quant_v_temp
    """
    # Internal intermediates (not exposed to caller).
    k_norm_buf = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.BF16)
    v_norm_buf = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.BF16)

    for ki_chunk in pl.parallel(0, NUM_KV_HEADS, 8):
        # Scope A: K RoPE + L2 norm + normalize.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_k_norm"):
            for ki in pl.range(ki_chunk, ki_chunk + 8):
                kv_col = ki * HEAD_DIM

                k_lo = pl.slice(k_proj_tile, [TOK_TILE, HALF_DIM], [0, kv_col])
                k_hi = pl.slice(k_proj_tile, [TOK_TILE, HALF_DIM],
                                [0, kv_col + HALF_DIM])
                rot_lo = pl.sub(
                    pl.mul(k_lo, cos_lo_all),
                    pl.mul(k_hi, sin_lo_all),
                )
                rot_hi = pl.add(
                    pl.mul(k_hi, cos_hi_all),
                    pl.mul(k_lo, sin_hi_all),
                )

                sq_lo = pl.reshape(
                    pl.row_sum(pl.mul(rot_lo, rot_lo)), [TOK_TILE, 1],
                )
                sq_hi = pl.reshape(
                    pl.row_sum(pl.mul(rot_hi, rot_hi)), [TOK_TILE, 1],
                )
                k_sq_sum = pl.add(sq_lo, sq_hi)
                k_l2_norm = pl.sqrt(pl.add(k_sq_sum, EPS))

                inv_norm = pl.recip(k_l2_norm)
                k_norm_lo = pl.row_expand_mul(rot_lo, inv_norm)
                k_norm_hi = pl.row_expand_mul(rot_hi, inv_norm)
                k_norm_lo_bf16 = pl.cast(k_norm_lo, target_type=pl.BF16)
                k_norm_hi_bf16 = pl.cast(k_norm_hi, target_type=pl.BF16)
                k_norm_buf = pl.assemble(
                    k_norm_buf, k_norm_lo_bf16, [0, kv_col],
                )
                k_norm_buf = pl.assemble(
                    k_norm_buf, k_norm_hi_bf16, [0, kv_col + HALF_DIM],
                )

                # Per-row write: k_l2_norm is [TOK_TILE, 1] (ColMajor after
                # reshape), pl.assemble to a narrow column in a RowMajor
                # buffer only fills row 0.  Flatten and write per-row instead.
                k_l2_norm_flat = pl.reshape(k_l2_norm, [TOK_TILE])
                for _ti in pl.range(TOK_TILE):
                    pl.write(k_scales_buf, [_ti, ki],
                                            pl.read(k_l2_norm_flat, [_ti]))

        # Scope B: K rotate + quantize.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_rotate_quant", optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=1)]):
            for ki in pl.range(ki_chunk, ki_chunk + 8):
                kv_col = ki * HEAD_DIM
                k_norm_bf16 = pl.slice(
                    k_norm_buf, [TOK_TILE, HEAD_DIM], [0, kv_col],
                )
                k_rot = pl.matmul(k_norm_bf16, rot_matrix, out_dtype=pl.FP32)

                # Lloyd-Max boundary search: idx = #{b : k_rot >= b} in [0, 15].
                # Compare against the SCALAR boundary _bN, not pl.full(...). A
                # tile-vs-scalar cmp lowers to tile.cmps -> TCMPS, whose CPU-sim
                # impl is correct; a tile-vs-tile cmp (pl.full rhs) lowers to
                # tile.cmp -> TCMP, whose CPU-sim impl can't model the FP32-src
                # -> UINT8-mask dst and fails to compile on a2a3sim. Numerically
                # identical; keep the scalar form (device is fine either way).
                k_idx = pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b0, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b1, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b2, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b3, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b4, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b5, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b6, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b7, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b8, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b9, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b10, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b11, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b12, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b13, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b14, cmp_type=5))
                # Pack two 4-bit indices per UINT8 (half/half layout):
                # byte c = idx[c+HALF_DIM]*16 + idx[c], c in 0..HALF_DIM-1.
                # Arithmetic (not bitwise): shls/and_/or_ are tile-only ops and
                # reject GM-slice tensors; unified mul/add accept them.
                k_lo = pl.slice(k_idx, [TOK_TILE, HALF_DIM], [0, 0])
                k_hi = pl.slice(k_idx, [TOK_TILE, HALF_DIM], [0, HALF_DIM])
                k_packed = pl.add(pl.mul(k_hi, 16.0), k_lo)            # FP32, 0..255 (exact)
                k_packed = pl.set_validshape(k_packed, TOK_TILE, HALF_DIM)
                # A2/A3 has no tile cast path into UINT8 with a dynamic valid
                # shape.  Scalar writes use the native scalar conversion and
                # preserve the exact packed byte value.
                k_sign_mask = pl.cmp(k_packed, 128.0, cmp_type=5)
                k_signed = pl.sub(
                    k_packed, pl.mul(k_sign_mask, 256.0),
                )
                k_signed = pl.set_validshape(k_signed, TOK_TILE, HALF_DIM)
                k_packed_half = pl.cast(k_signed, target_type=pl.FP16, mode="round")
                k_packed_half = pl.set_validshape(k_packed_half, TOK_TILE, HALF_DIM)
                k_packed_i8 = pl.cast(k_packed_half, target_type=pl.INT8, mode="trunc")
                k_packed_i8 = pl.set_validshape(k_packed_i8, TOK_TILE, HALF_DIM)
                k_packed_u8 = pl.reinterpret_view(k_packed_i8, pl.UINT8)
                quant_k_temp = pl.assemble(quant_k_temp, k_packed_u8, [0, ki * HALF_DIM])

        # Scope C: V L2 norm + normalize.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_norm"):
            for ki in pl.range(ki_chunk, ki_chunk + 8):
                kv_col = ki * HEAD_DIM

                v_all = pl.slice(v_proj_tile, [TOK_TILE, HEAD_DIM],
                                 [0, kv_col])
                v_sq_sum = pl.reshape(
                    pl.row_sum(pl.mul(v_all, v_all)), [TOK_TILE, 1],
                )
                v_l2_norm = pl.sqrt(pl.add(v_sq_sum, EPS))
                v_norm = pl.row_expand_mul(v_all, pl.recip(v_l2_norm))
                v_norm_bf16 = pl.cast(v_norm, target_type=pl.BF16)
                v_norm_buf = pl.assemble(v_norm_buf, v_norm_bf16, [0, kv_col])

                v_l2_norm_flat = pl.reshape(v_l2_norm, [TOK_TILE])
                for _ti in pl.range(TOK_TILE):
                    pl.write(v_scales_buf, [_ti, ki],
                                             pl.read(v_l2_norm_flat, [_ti]))

        # Scope D: V rotate + quantize.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_rotate_quant", optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=1)]):
            for ki in pl.range(ki_chunk, ki_chunk + 8):
                kv_col = ki * HEAD_DIM
                v_norm_bf16 = pl.slice(
                    v_norm_buf, [TOK_TILE, HEAD_DIM], [0, kv_col],
                )
                v_rot = pl.matmul(v_norm_bf16, rot_matrix, out_dtype=pl.FP32)

                # Boundary search (see Scope B): compare against scalar _bN.
                v_idx = pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b0, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b1, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b2, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b3, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b4, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b5, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b6, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b7, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b8, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b9, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b10, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b11, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b12, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b13, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b14, cmp_type=5))
                # Pack two 4-bit indices per UINT8 (half/half layout, see Scope B).
                v_lo = pl.slice(v_idx, [TOK_TILE, HALF_DIM], [0, 0])
                v_hi = pl.slice(v_idx, [TOK_TILE, HALF_DIM], [0, HALF_DIM])
                v_packed = pl.add(pl.mul(v_hi, 16.0), v_lo)
                v_packed = pl.set_validshape(v_packed, TOK_TILE, HALF_DIM)
                v_sign_mask = pl.cmp(v_packed, 128.0, cmp_type=5)
                v_signed = pl.sub(
                    v_packed, pl.mul(v_sign_mask, 256.0),
                )
                v_signed = pl.set_validshape(v_signed, TOK_TILE, HALF_DIM)
                v_packed_half = pl.cast(v_signed, target_type=pl.FP16, mode="round")
                v_packed_half = pl.set_validshape(v_packed_half, TOK_TILE, HALF_DIM)
                v_packed_i8 = pl.cast(v_packed_half, target_type=pl.INT8, mode="trunc")
                v_packed_i8 = pl.set_validshape(v_packed_i8, TOK_TILE, HALF_DIM)
                v_packed_u8 = pl.reinterpret_view(v_packed_i8, pl.UINT8)
                quant_v_temp = pl.assemble(quant_v_temp, v_packed_u8, [0, ki * HALF_DIM])
