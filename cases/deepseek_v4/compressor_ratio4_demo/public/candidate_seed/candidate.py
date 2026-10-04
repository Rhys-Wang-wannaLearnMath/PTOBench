# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=4 overlap).

Uses overlapping state layout with 8 slots.
Front slots 0-3 at columns [0:HEAD_DIM], back slots 4-7 at columns [HEAD_DIM:OUT_DIM].
Tree reduction for softmax+pool. State shift after compression."""


import pypto.language as pl

from config import B, COMPRESS_RATIO, D, EPS, FP32_NEG_INF, HEAD_DIM, HEAD_DIM_INV, IDX_KV_LEN, NOPE_HEAD_DIM, OUT_DIM, ROPE_HEAD_DIM, S, STATE_LEN

# tiling
ROPE_CHUCK = 32
K_CHUNK = 512
OUT_CHUNK = 64   # keep RHS tiles <=64KB on current ptoas/a2a3
HEAD_CHUNK = 64 if B * S >= 64 else 128
HEAD_DIM_CHUCK = 128
MM_B_TILE = 16          # cube matmul rows must be 16-aligned on current ptoas
B_PAD = ((B + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
assert S == 1, "standalone decode compressor_ratio4 assumes DECODE_SEQ == 1"
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK


@pl.jit.inline
def compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    x_flat = pl.reshape(x, [B * S, D])
    cmp4_kv_proj_scratch = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_scratch = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    ape_row = pl.cast(start_pos % COMPRESS_RATIO, target_type=pl.INDEX)
    pre_tokens = COMPRESS_RATIO - (start_pos % COMPRESS_RATIO)
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])

    # Output TensorSpec init values are not guaranteed to survive the current
    # runner path. Recreate golden's -inf score_state tail before scattering
    # active rows.
    with pl.spmd(B * STATE_LEN * (OUT_DIM // OUT_CHUNK), name_hint="score_state_init") as init_tid:
        init_idx = pl.tile.get_block_idx()
        init_chunks_per_batch = STATE_LEN * (OUT_DIM // OUT_CHUNK)
        init_b = init_idx // init_chunks_per_batch
        init_chunk = init_idx - init_b * init_chunks_per_batch
        init_col = init_chunk * OUT_CHUNK
        score_state_flat[init_b : init_b + 1, init_col : init_col + OUT_CHUNK] = \
            pl.full([1, OUT_CHUNK], dtype=pl.FP32, value=FP32_NEG_INF)

    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_score_proj"):
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_CHUNK], [0, 0], valid_shape=[B * S, K_CHUNK])
            wkv_tile = wkv[0 : K_CHUNK, o0 : o0 + OUT_CHUNK]
            wgate_tile = wgate[0 : K_CHUNK, o0 : o0 + OUT_CHUNK]
            kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
            score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)

            for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                x_tile = pl.slice(x_flat, [MM_B_TILE, K_CHUNK], [0, k0], valid_shape=[B * S, K_CHUNK])
                wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

            cmp4_kv_proj_scratch = pl.assemble(cmp4_kv_proj_scratch, kv_acc, [0, o0])
            cmp4_score_proj_scratch = pl.assemble(cmp4_score_proj_scratch, score_acc, [0, o0])

        scatter_stop = S
        if pre_tokens < S:
            scatter_stop = pre_tokens
        for s in pl.range(scatter_stop):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter", deps=[init_tid]):
                kv_tile = cmp4_kv_proj_scratch[0:B, o0 : o0 + OUT_CHUNK]
                score_tile = cmp4_score_proj_scratch[0:B, o0 : o0 + OUT_CHUNK]
                token_ape_row = (ape_row + s) % COMPRESS_RATIO
                ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                ape_base = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                slot_col0_s = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, slot_col0_s + o0])
                score_state_flat = pl.assemble(score_state_flat, score_tile, [0, slot_col0_s + o0])

    # Pool / norm / rope buffers are per-batch (one compressed kv per batch when
    # compression fires); kv output is still [B, S, HEAD_DIM] with only the
    # b,0,: row populated, matching attention_csa's read of cmp_out[b, 0, :].
    # kv_flat = reshape(kv, [B*S, HEAD_DIM]) maps kv[b, s, :] -> row b*S + s.
    pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.BF16)
    kv_final = pl.create_tensor([B_PAD, HEAD_DIM], dtype=pl.FP32)
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])

    if (start_pos % COMPRESS_RATIO) + S >= COMPRESS_RATIO:
        for hb in pl.parallel(0, HEAD_BLOCKS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool"):
                h0 = hb * HEAD_CHUNK
                last_col0 = (STATE_LEN - 1) * OUT_DIM + HEAD_DIM + h0
                mi = score_state_flat[:, last_col0 : last_col0 + HEAD_CHUNK]
                li = pl.exp(pl.sub(mi, mi))
                oi = kv_state_flat[:, last_col0 : last_col0 + HEAD_CHUNK]

                for s in pl.range(0, COMPRESS_RATIO):
                    front_col0 = s * OUT_DIM + h0
                    front_score = score_state_flat[:, front_col0 : front_col0 + HEAD_CHUNK]
                    front_kv = kv_state_flat[:, front_col0 : front_col0 + HEAD_CHUNK]
                    mi_next_front = pl.maximum(mi, front_score)
                    alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                    beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                    li = pl.add(pl.mul(alpha_front, li), beta_front)
                    oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                    mi = mi_next_front

                for s in pl.range(COMPRESS_RATIO, STATE_LEN - 1):
                    back_col0 = s * OUT_DIM + HEAD_DIM + h0
                    back_score = score_state_flat[:, back_col0 : back_col0 + HEAD_CHUNK]
                    back_kv = kv_state_flat[:, back_col0 : back_col0 + HEAD_CHUNK]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_kv = pl.assemble(pooled_kv, pooled_chunk, [0, h0])

        for s in pl.range(0, COMPRESS_RATIO, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_shift"):
                src_col0 = (COMPRESS_RATIO + s) * OUT_DIM
                dst_col0 = s * OUT_DIM
                for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                    dep_col0 = o0 % HEAD_DIM
                    dep_tile = pooled_kv[:, dep_col0 : dep_col0 + OUT_CHUNK]
                    dep_zero = pl.sub(dep_tile, dep_tile)
                    kv_tile = kv_state_flat[:, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                    score_tile = score_state_flat[:, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                    kv_tile = pl.add(kv_tile, dep_zero)
                    score_tile = pl.add(score_tile, dep_zero)
                    kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, dst_col0 + o0])
                    score_state_flat = pl.assemble(score_state_flat, score_tile, [0, dst_col0 + o0])

        norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
            partial_sq = pl.full([1, B], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
                # Golden applies rmsnorm to kv.to(torch.bfloat16), then casts to FP32 inside rmsnorm.
                kv_rms_chunk = pl.cast(
                    pl.cast(pooled_kv[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(kv_rms_chunk, kv_rms_chunk)), [1, B]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [B, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
                kv_norm_chunk = pl.cast(
                    pl.cast(pooled_kv[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                gamma = norm_w_2d[:, k0 : k0 + HEAD_CHUNK]
                normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                normed_kv = pl.assemble(normed_kv, pl.cast(normed_chunk, target_type=pl.BF16, mode="rint"), [0, k0])

        kv_rope = pl.create_tensor([B_PAD, ROPE_HEAD_DIM], dtype=pl.BF16)
        kv_proj_even = pl.create_tensor([B_PAD, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        kv_proj_odd = pl.create_tensor([B_PAD, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        rope_even = pl.create_tensor([B_PAD, ROPE_HEAD_DIM // 2], dtype=pl.BF16)
        rope_odd = pl.create_tensor([B_PAD, ROPE_HEAD_DIM // 2], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_slice"):
            kv_rope = pl.assemble(kv_rope, normed_kv[0:B, NOPE_HEAD_DIM : HEAD_DIM], [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_slice"):
            kv_rope_tile = pl.slice(kv_rope, [MM_B_TILE, ROPE_CHUCK], [0, 0], valid_shape=[B, ROPE_CHUCK])
            even_select_tile = even_select[0 : ROPE_CHUCK, :]
            odd_select_tile = odd_select[0 : ROPE_CHUCK, :]
            even_acc = pl.matmul(kv_rope_tile, even_select_tile, out_dtype=pl.FP32)
            odd_acc = pl.matmul(kv_rope_tile, odd_select_tile, out_dtype=pl.FP32)

            for r0 in pl.range(ROPE_CHUCK, ROPE_HEAD_DIM, ROPE_CHUCK):
                kv_rope_tile = pl.slice(kv_rope, [MM_B_TILE, ROPE_CHUCK], [0, r0], valid_shape=[B, ROPE_CHUCK])
                even_select_tile = even_select[r0 : r0 + ROPE_CHUCK, :]
                odd_select_tile = odd_select[r0 : r0 + ROPE_CHUCK, :]
                even_acc = pl.matmul_acc(even_acc, kv_rope_tile, even_select_tile)
                odd_acc = pl.matmul_acc(odd_acc, kv_rope_tile, odd_select_tile)
            kv_proj_even = pl.assemble(kv_proj_even, even_acc, [0, 0])
            kv_proj_odd = pl.assemble(kv_proj_odd, odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_apply"):
            even_tile = kv_proj_even[0:B_PAD, :]
            odd_tile = kv_proj_odd[0:B_PAD, :]
            rope_even_acc = pl.cast(pl.sub(pl.col_expand_mul(even_tile, cos), pl.col_expand_mul(odd_tile, sin)), target_type=pl.BF16, mode="rint")
            rope_odd_acc = pl.cast(pl.add(pl.col_expand_mul(even_tile, sin), pl.col_expand_mul(odd_tile, cos)), target_type=pl.BF16, mode="rint")
            rope_even = pl.assemble(rope_even, rope_even_acc, [0, 0])
            rope_odd = pl.assemble(rope_odd, rope_odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_assemble"):
            rope_even_tile = rope_even[0:B_PAD, 0 : ROPE_CHUCK]
            rope_odd_tile = rope_odd[0:B_PAD, 0 : ROPE_CHUCK]
            even_select_tile_t = even_select[:, 0 : ROPE_CHUCK]
            odd_select_tile_t = odd_select[:, 0 : ROPE_CHUCK]
            rope_acc = pl.matmul(rope_even_tile, even_select_tile_t, out_dtype=pl.FP32, b_trans=True)
            rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

            for r0 in pl.range(ROPE_CHUCK, ROPE_HEAD_DIM // 2, ROPE_CHUCK):
                rope_even_tile = rope_even[0:B_PAD, r0 : r0 + ROPE_CHUCK]
                rope_odd_tile = rope_odd[0:B_PAD, r0 : r0 + ROPE_CHUCK]
                even_select_tile_t = even_select[:, r0 : r0 + ROPE_CHUCK]
                odd_select_tile_t = odd_select[:, r0 : r0 + ROPE_CHUCK]
                rope_acc = pl.matmul_acc(rope_acc, rope_even_tile, even_select_tile_t, b_trans=True)
                rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_write"):
            normed_kv = pl.assemble(normed_kv, pl.cast(rope_acc[0:B, 0:ROPE_HEAD_DIM], target_type=pl.BF16, mode="rint"), [0, NOPE_HEAD_DIM])

        # Build a [B, HEAD_DIM] FP32 kv_final tile (hadamard-rotated or plain).
        if rotate:
            for o0 in pl.range(0, HEAD_DIM, OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
                    kv_proj_tile = pl.slice(normed_kv, [MM_B_TILE, HEAD_DIM], [0, 0], valid_shape=[B, HEAD_DIM])
                    hadamard_tile = hadamard[0 : HEAD_DIM, o0 : o0 + OUT_CHUNK]
                    kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
                    kv_final = pl.assemble(kv_final, kv_hadamard_acc, [0, o0])
        else:
            for o0 in pl.parallel(0, HEAD_DIM, OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_write"):
                    kv_out_tile = normed_kv[:, o0 : o0 + OUT_CHUNK]
                    kv_final = pl.assemble(kv_final, pl.cast(kv_out_tile, target_type=pl.FP32), [0, o0])

        # Per-batch fan-out: write kv_final[b] to kv[b, 0, :] (row b*S of kv_flat)
        # and to kv_cache[b, cache_col, :]. With S>=2, kv[b, 1:, :] stays at its
        # initial value (zero-init in the test fixture; garbage on hot path —
        # downstream attention only reads kv[b, 0, :]).
        kv_cache_flat = pl.reshape(kv_cache, [B * IDX_KV_LEN, HEAD_DIM])
        cache_col = start_pos // COMPRESS_RATIO
        for b_idx in pl.parallel(B):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write"):
                kv_row_fp32 = kv_final[b_idx : b_idx + 1, 0 : HEAD_DIM]
                kv_flat = pl.assemble(kv_flat, kv_row_fp32, [b_idx * S, 0])
                cache_row = b_idx * IDX_KV_LEN + cache_col
                kv_cache_flat = pl.assemble(
                    kv_cache_flat,
                    pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint"),
                    [cache_row, 0],
                )
        if pre_tokens < S:
            for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                for s in pl.range(pre_tokens, S):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_next"):
                        kv_tile = cmp4_kv_proj_scratch[0:B, o0 : o0 + OUT_CHUNK]
                        score_tile = cmp4_score_proj_scratch[0:B, o0 : o0 + OUT_CHUNK]
                        shift_dep_col0 = (COMPRESS_RATIO - 1) * OUT_DIM + o0
                        dep_zero = pl.sub(
                            kv_state_flat[:, shift_dep_col0 : shift_dep_col0 + OUT_CHUNK],
                            kv_state_flat[:, shift_dep_col0 : shift_dep_col0 + OUT_CHUNK],
                        )
                        kv_tile = pl.add(kv_tile, dep_zero)
                        score_tile = pl.add(score_tile, dep_zero)
                        token_ape_row = (ape_row + s) % COMPRESS_RATIO
                        ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                        ape_base = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                        slot_col0_s = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                        kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, slot_col0_s + o0])
                        score_state_flat = pl.assemble(score_state_flat, score_tile, [0, slot_col0_s + o0])

        for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
            for s in pl.range(pre_tokens):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_shift_boundary"):
                    token_ape_row = (ape_row + s) % COMPRESS_RATIO
                    src_col0 = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                    dst_col0 = token_ape_row * OUT_DIM
                    kv_tile = kv_state_flat[:, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                    score_tile = score_state_flat[:, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                    kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, dst_col0 + o0])
                    score_state_flat = pl.assemble(score_state_flat, score_tile, [0, dst_col0 + o0])

        kv_cache = pl.reshape(kv_cache_flat, [B, IDX_KV_LEN, HEAD_DIM])

    kv_state = pl.reshape(kv_state_flat, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [B, STATE_LEN, OUT_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv, kv_state, score_state, kv_cache


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    kv_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    score_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    kv, kv_state, score_state, kv_cache = compressor(
        x, kv, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin, even_select, odd_select, hadamard, kv_cache, start_pos, rotate
    )
    return kv, kv_state, score_state, kv_cache
