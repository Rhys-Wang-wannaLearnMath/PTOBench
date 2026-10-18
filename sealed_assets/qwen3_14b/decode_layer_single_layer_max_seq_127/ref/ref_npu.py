"""Certified NPU Ref for qwen3_14b.decode_layer_single_layer_max_seq_127.

This closure preserves the selected official decode topology and applies only
current-interface bindings: fixed values come from the public Case config, the
chunk dispatch is bound to the selected one-layer contract, and the stale
source geometry assertion is omitted because QKV_K_CHUNKS is intentionally 5.
"""

import pypto.language as pl

from config import (
    ATTN_SCALE, BATCH, BLOCK_SIZE, CACHE_ROWS, EPS, HEAD_DIM, HEAD_DIM_INV,
    HIDDEN, HIDDEN_INV, INTERMEDIATE, KV_CACHE_ROWS_DYN, KV_HIDDEN, HALF_DIM,
    MAX_BLOCKS_PER_SEQ, MAX_CTX_BLOCKS, MAX_SEQ, NUM_HEADS, NUM_KV_HEADS,
    NUM_PAGES, Q_HEAD_BATCH, Q_HEAD_PAD, Q_PER_KV, SEQ_TILE,
)

# Implementation tuning constants preserved from the selected official source.
RMSNORM_K_CHUNK = 256
XG_BLOCKS = 5
TM = 16
TN = 256
TK = 256
QKV_N_TILE = 512
N_SUB = QKV_N_TILE // TN
Q_ON = HIDDEN // QKV_N_TILE
KV_ON = KV_HIDDEN // QKV_N_TILE
QKV_OK = 5
QKV_K_SLICE = HIDDEN // QKV_OK
QKV_K_CHUNKS = QKV_K_SLICE // TK
TOKENS_PER_SPLIT = SEQ_TILE
BLOCKS_PER_SPLIT = TOKENS_PER_SPLIT // SEQ_TILE
KV_SPLITS = MAX_CTX_BLOCKS // BLOCKS_PER_SPLIT
GP_SIZE = 8
HEAD_GROUPS = NUM_KV_HEADS // GP_SIZE
NUM_CORES = 24
FA_TABLE_CAP = BATCH * MAX_CTX_BLOCKS
OS_WORK = BATCH * NUM_KV_HEADS
HEADS_PER_ROPE = 4
NUM_ROPE_GROUPS = NUM_KV_HEADS // HEADS_PER_ROPE
ROPE_GROUP_CORES = 16
ROPE_GROUP_ITEMS_PER_CORE = (HEADS_PER_ROPE * BATCH) // ROPE_GROUP_CORES
ROPE_CORES = 32
ROPE_ITEMS_PER_CORE = (NUM_KV_HEADS * BATCH) // ROPE_CORES
K_SPLITS_OUT = 5
N_SPLITS_OUT = 10
OUT_INNER_TK = 64
OUT_TN = HIDDEN // N_SPLITS_OUT
OUT_TK = HIDDEN // K_SPLITS_OUT
OUT_N_SUB_K = OUT_TK // OUT_INNER_TK
K_CHUNK = 256
MLP_TN = 1024
K_SPLITS_MLP = 5
MLP_INNER_TK = 64
MLP_K_SLICE = HIDDEN // K_SPLITS_MLP
MLP_N_SUB_K = MLP_K_SLICE // MLP_INNER_TK
MLP_ON = INTERMEDIATE // MLP_TN
MLP_OUT_CHUNK = 256
SILU_INNER_CHUNKS = MLP_TN // MLP_OUT_CHUNK
DOWN_TN = 1024
DOWN_TK = 64
DOWN_ON = HIDDEN // DOWN_TN
K_SPLITS = INTERMEDIATE // DOWN_TN
N_SUB_K = DOWN_TN // DOWN_TK
N_PER_CAST_K = MLP_K_SLICE // OUT_TN

assert QKV_N_TILE % TN == 0
assert HIDDEN % QKV_N_TILE == 0 and KV_HIDDEN % QKV_N_TILE == 0
assert HIDDEN % QKV_OK == 0
assert QKV_K_SLICE % TK == 0
assert Q_ON == 10 and KV_ON == 2
assert TM == BATCH and N_SUB == 2 and QKV_K_CHUNKS == 4
assert TOKENS_PER_SPLIT % SEQ_TILE == 0
assert MAX_CTX_BLOCKS % BLOCKS_PER_SPLIT == 0
assert KV_SPLITS * BLOCKS_PER_SPLIT == MAX_CTX_BLOCKS
assert NUM_KV_HEADS % GP_SIZE == 0
assert HEAD_GROUPS * GP_SIZE == NUM_KV_HEADS
assert GP_SIZE == NUM_KV_HEADS
assert FA_TABLE_CAP == BATCH * MAX_CTX_BLOCKS
assert DOWN_TN % MLP_OUT_CHUNK == 0
assert DOWN_ON * DOWN_TN == HIDDEN
assert K_SPLITS * DOWN_TN == INTERMEDIATE
assert MLP_ON * MLP_TN == INTERMEDIATE
assert K_SPLITS_MLP * MLP_K_SLICE == HIDDEN
assert MLP_TN % MLP_OUT_CHUNK == 0
assert MLP_TN == DOWN_TN
assert N_SPLITS_OUT * OUT_TN == HIDDEN
assert K_SPLITS_OUT * OUT_TK == HIDDEN
assert OUT_N_SUB_K * OUT_INNER_TK == OUT_TK
assert N_PER_CAST_K * OUT_TN == MLP_K_SLICE
assert (NUM_KV_HEADS * BATCH) % ROPE_CORES == 0

@pl.jit.inline
def _decode_layer(  # noqa: PLR0913 — model signature is intrinsic
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.FP32],  # FP32: inter-layer carry (was BF16)
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Tensor[[BATCH, HIDDEN], pl.FP32],  # FP32: inter-layer carry (was BF16)
    # normed_in: THIS layer's x*gamma (BF16), produced by the previous layer's fused
    # dcr_xgamma (x_gamma_0 for layer 0). Consumed by QKV — replaces the local x_gamma.
    normed_in: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    # normed_out: NEXT layer's x*gamma, written by THIS layer's dcr_xgamma. Escapes via
    # the inline alias (verified: inline out-param tensor writes are visible to caller).
    normed_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
    next_gamma_idx: pl.Scalar[pl.INT32],  # clamped min(layer_idx+1, N-1) for dcr_xgamma's gamma
    # Cross-iteration carries: prev_out_tids = DOWN_ON slab writers of `hidden_states`
    # (prev dcr / copy_hidden) — gate rms_recip / residual / seeds. prev_normed_tids =
    # DOWN_ON slab writers of `normed_in` (prev dcr_xgamma / x_gamma_0) — gate QKV. Both
    # refilled by this layer's dcr_xgamma (same task writes out + normed_out).
    prev_out_tids: pl.Array[DOWN_ON, pl.TASK_ID],
    prev_normed_tids: pl.Array[DOWN_ON, pl.TASK_ID],
) -> pl.Tensor[[BATCH, HIDDEN], pl.FP32]:
    # Per-layer offsets into the STACKED weights / PAGED KV cache. decode_fwd passes
    # the running loop index 0.._FWD_NLAYERS-1; decode_fwd_layers passes
    # 0.._CHUNK_NLAYERS-1 (per-chunk weight slices).
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    # Paged KV: rows are runtime-dynamic (the paged pool sizes them). Derive the
    # per-layer stride and the block-table row stride from the tensor dims, exactly
    # as prefill_fwd does, so decode reads the SAME pool prefill wrote.
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    layer_cache_base = layer_idx * layer_cache_rows
    user_batch = pl.tensor.dim(seq_lens, 0)
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // user_batch
    q_norm_w = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
    k_norm_w = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])

    # Scope 1
    # down_proj TaskIds — HOISTED to orchestration scope (declared before
    # manual_scope) so the consolidated `down_cast_residual` writer that runs
    # AFTER the manual_scope can gate on them via deps=. Filled inside the
    # manual_scope down_proj loop; the consolidated writer reads them per-index
    # (deps=[down_tids[k] for k in range(DOWN_ON * K_SPLITS)] — list-comprehension
    # per-index fence works for large N; whole-array deps=[down_tids] does not).
    down_tids = pl.array.create(DOWN_ON * K_SPLITS, pl.TASK_ID)
    q_tile_tids = pl.array.create(Q_ON * QKV_OK, pl.TASK_ID)
    k_tile_tids = pl.array.create(KV_ON * QKV_OK, pl.TASK_ID)
    v_tile_tids = pl.array.create(KV_ON * QKV_OK, pl.TASK_ID)
    qk_tids = pl.array.create(NUM_KV_HEADS, pl.TASK_ID)  # fused qk_norm (gamma+recip)
    rope_grp_tids = pl.array.create(NUM_ROPE_GROUPS, pl.TASK_ID)
    inv_rms_states = pl.create_tensor([BATCH, 1], dtype=pl.FP32)  # deferred 1/rms denominator
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    # ── Scope 1: input RMSNorm, SPLIT into two INDEPENDENT steps. ──
    # RMSNorm(x) = (x * inv_rms) * gamma, where inv_rms[b] = 1/sqrt(mean_k x^2 +
    # eps) is a per-row SCALAR. Q/K/V proj and RoPE are linear, so the 1/rms factor
    # commutes through them: q = inv_rms * ((x*gamma) @ Wq). We therefore DEFER the
    # 1/rms division past the projections and fold it into rope_qkv (one scalar mul
    # per batch row). This decouples the sum-of-squares reduction from the gamma
    # scaling: `x_gamma` (which feeds QKV) no longer waits on the reduction, and
    # `rms_recip` overlaps the QKV proj. normed_in is consumed ONLY by QKV; the
    # residual / post_rms path reads raw hidden_states, so it is unaffected.
    #
    # WHOLE-LAYER manual scope (x_gamma .. rope): tensormap registration is
    # suppressed; every cross-task edge below is an explicit deps=. x_gamma slab
    # n deps ONLY on prev_out_tids[n] (the previous layer's dcr writer of that
    # carry slab, copy_hidden for layer 0) — the fine-grained dcr->x_gamma edge
    # the auto path cannot express without whole-parent WAW serialization.
    # normed_in is now the `normed_in` PARAM (produced by prev dcr_xgamma / x_gamma_0).

    # ── Scope 2 allocations (hoisted before the manual scope). ──
    all_q_padded = pl.create_tensor([BATCH * NUM_KV_HEADS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    all_oi_tmp = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_CTX_BLOCKS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32
    )
    all_cur_mi = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_CTX_BLOCKS * Q_HEAD_PAD, 1], dtype=pl.FP32
    )
    all_cur_li = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_CTX_BLOCKS * Q_HEAD_PAD, 1], dtype=pl.FP32
    )
    # Block-level (dense) work list. `fa_work_table[w]` encodes the w-th REAL
    # seq-block as `b * MAX_CTX_BLOCKS + p`; `fa_total[0]` holds the number of
    # real blocks. Built once by the `fa_work_build` AIV task, consumed by
    # fa_fused's grid-stride. Sized to the worst case (every sequence full).
    fa_work_table = pl.create_tensor([FA_TABLE_CAP, 1], dtype=pl.INT32)
    fa_total = pl.create_tensor([1, 1], dtype=pl.INT32)

    with pl.manual_scope():
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="rms_recip",
            deps=[prev_out_tids[i] for i in range(DOWN_ON)],
        ) as rms_tid:
            partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):
                k0 = kb * RMSNORM_K_CHUNK
                x_chunk = hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK]  # FP32 already (was cast from BF16)
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]),
                )
            variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            inv_rms_states = pl.assemble(inv_rms_states, inv_rms, [0, 0])

        # ── Scope 1: Q projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed") as q_seed_tid:  # no explicit dep: runtime q_proj WAR hazard orders it after prev qk_norm
            for snb in pl.pipeline(Q_ON, stage=2):
                q_proj = pl.assemble(
                    q_proj, pl.full([BATCH, QKV_N_TILE], dtype=pl.FP32, value=0.0), [0, snb * QKV_N_TILE]
                )
        for q_nt in pl.parallel(Q_ON):
            q_n_region = q_nt * QKV_N_TILE
            for q_ks in pl.range(QKV_OK):
                q_k_base = q_ks * QKV_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="q_proj",
                    deps=[prev_normed_tids[q_ks], q_seed_tid],  # 1:1 with normed slab (QKV_K_SLICE=DOWN_TN)
                ) as q_tid:
                    for n_sub in pl.range(N_SUB):
                        n0 = q_n_region + n_sub * TN
                        q_acc = pl.matmul(
                            normed_in[:, q_k_base : q_k_base + TK],
                            wq[layer_hidden_base + q_k_base : layer_hidden_base + q_k_base + TK, n0 : n0 + TN],
                            out_dtype=pl.FP32,
                        )
                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                            kk = q_k_base + kc * TK
                            q_acc = pl.matmul_acc(
                                q_acc, normed_in[:, kk : kk + TK], wq[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                            )
                        q_proj = pl.assemble(q_proj, q_acc, [0, n0], atomic=pl.AtomicType.Add)
                q_tile_tids[q_nt * QKV_OK + q_ks] = q_tid

        # ── Scope 1: K projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as k_seed_tid:
            k_proj = pl.assemble(k_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])
        for k_nt in pl.parallel(KV_ON):
            k_n_region = k_nt * QKV_N_TILE
            for k_ks in pl.range(QKV_OK):
                k_k_base = k_ks * QKV_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="k_proj",
                    deps=[prev_normed_tids[0], prev_normed_tids[1], prev_normed_tids[2], prev_normed_tids[3], prev_normed_tids[4], k_seed_tid],
                ) as k_tid:
                    for n_sub in pl.range(N_SUB):
                        n0 = k_n_region + n_sub * TN
                        k_acc = pl.matmul(
                            normed_in[:, k_k_base : k_k_base + TK],
                            wk[layer_hidden_base + k_k_base : layer_hidden_base + k_k_base + TK, n0 : n0 + TN],
                            out_dtype=pl.FP32,
                        )
                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                            kk = k_k_base + kc * TK
                            k_acc = pl.matmul_acc(
                                k_acc, normed_in[:, kk : kk + TK], wk[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                            )
                        k_proj = pl.assemble(k_proj, k_acc, [0, n0], atomic=pl.AtomicType.Add)
                k_tile_tids[k_nt * QKV_OK + k_ks] = k_tid

        # ── Scope 1: V projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as v_seed_tid:
            v_proj = pl.assemble(v_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])
        for v_nt in pl.parallel(KV_ON):
            v_n_region = v_nt * QKV_N_TILE
            for v_ks in pl.range(QKV_OK):
                v_k_base = v_ks * QKV_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="v_proj",
                    deps=[prev_normed_tids[0], prev_normed_tids[1], prev_normed_tids[2], prev_normed_tids[3], prev_normed_tids[4], v_seed_tid],
                ) as v_tid:
                    for n_sub in pl.range(N_SUB):
                        n0 = v_n_region + n_sub * TN
                        v_acc = pl.matmul(
                            normed_in[:, v_k_base : v_k_base + TK],
                            wv[layer_hidden_base + v_k_base : layer_hidden_base + v_k_base + TK, n0 : n0 + TN],
                            out_dtype=pl.FP32,
                        )
                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                            kk = v_k_base + kc * TK
                            v_acc = pl.matmul_acc(
                                v_acc, normed_in[:, kk : kk + TK], wv[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                            )
                        v_proj = pl.assemble(v_proj, v_acc, [0, n0], atomic=pl.AtomicType.Add)
                v_tile_tids[v_nt * QKV_OK + v_ks] = v_tid

        # ── Scope 2 prep: build the dense block-level work list on an AIV task. ──
        # Inputs are external (seq_lens) — no deps.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_work_build") as work_tid:
            cursor = pl.read(seq_lens, [0]) * 0  # scalar 0 (INDEX)
            for wb in pl.unroll(BATCH):
                wb_ctx = (pl.read(seq_lens, [wb]) + (SEQ_TILE - 1)) // SEQ_TILE
                for wp in pl.range(wb_ctx):
                    # fa_work_table is INT32 (fixed-width for the host↔ptoas ABI; see
                    # KNOWN_ISSUES "pl.INDEX GM tensor width mismatch") — cast the
                    # index-typed encoding to match the tensor dtype.
                    pl.tensor.write(
                        fa_work_table, [cursor + wp, 0], pl.cast(wb * MAX_CTX_BLOCKS + wp, target_type=pl.INT32)
                    )
                cursor = cursor + wb_ctx
            pl.tensor.write(fa_total, [0, 0], pl.cast(cursor, target_type=pl.INT32))

        # ── Scope 2: per-head Qwen3 QK-norm, SPLIT into two INDEPENDENT steps. ──
        # Same trick as the input RMSNorm above. QKnorm(x)_head = (x * qk_inv_head) *
        # gamma, where qk_inv_head[b,h] = 1/sqrt(mean_d x^2 + eps) is a per-(row, head)
        # SCALAR. RoPE is linear within a head, so qk_inv commutes through it. We
        # therefore (a) apply ONLY the elementwise `* gamma` here — q_proj_norm /
        # k_proj_norm no longer wait on the reduction — and (b) DEFER the per-head
        # qk_inv reciprocal past RoPE, folding it into rope_qkv as one scalar mul per
        # head-row. The two steps read q_proj / k_proj independently, so `qk_recip`
        # (the sum-of-squares reduction) overlaps `qk_gamma` instead of serializing
        # after it.
        #
        # CONTROL EXPERIMENT: the deferred input-RMSNorm inv_rms is a POSITIVE per-row
        # scalar and QK-norm is scale-invariant, so it CANCELS inside this QK-norm and
        # the optimized path omits it on Q/K. Here we instead APPLY it explicitly — we
        # scale q_proj / k_proj by inv_rms[b] BEFORE the QK-norm in BOTH sub-steps
        # (qk_gamma AND qk_recip). Because the reciprocal step sees inv_rms*x its
        # denominator picks up a 1/inv_rms factor that exactly undoes the inv_rms in
        # the gamma step, so q_out / k_out are bit-for-bit the SAME as the optimized
        # path (RoPE folds the two together). This makes the full mathematical chain
        # input-RMSNorm -> proj -> QK-norm visible in the code, at the cost of two
        # redundant row-scales. Must be in BOTH steps; applying it to only one would
        # NOT cancel and would change the result.
        #
        # qk_inv layout is (head, batch[, q-in-group]) so rope_qkv can slice a
        # contiguous per-(KV head, batch) column. Per head h, the q reduction yields
        # [BATCH * Q_PER_KV, 1] rows ordered (b, j); we stack the NUM_KV_HEADS blocks,
        # so global row = h*BATCH*Q_PER_KV + b*Q_PER_KV + j.
        q_proj_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
        q_inv_states = pl.create_tensor([NUM_KV_HEADS * BATCH * Q_PER_KV, 1], dtype=pl.FP32)
        k_inv_states = pl.create_tensor([NUM_KV_HEADS * BATCH, 1], dtype=pl.FP32)

        # inv_rms[b] as a [BATCH, 1] column — applied row-wise to q_proj / k_proj
        # BEFORE QK-norm in both sub-steps below (the control-experiment scale).
        inv_rms_col = inv_rms_states[:, 0:1]

        # FUSED qk_norm PER KV-HEAD: gamma (q/k_proj_norm) AND the 1/rms reduction
        # (q/k_inv) in ONE task, reading each q/k tile from GM ONCE (qk_gamma + qk_recip
        # used to read the same data twice). Halves qk-stage GM reads and the qk task
        # count (16 -> 8). q_chunk/k_chunk (= inv_rms-scaled input) feed BOTH the gamma
        # assemble and the sum-of-squares. Each head gates on its 2 straddled q tiles +
        # 1 k tile. (recip is still deferred — folded into rope as a per-head scalar mul.)
        for h in pl.unroll(NUM_KV_HEADS):
            qt0 = (h * Q_PER_KV * HEAD_DIM) // QKV_N_TILE
            kt = (h * HEAD_DIM) // QKV_N_TILE
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="qk_norm",
                deps=[
                    q_tile_tids[qt0 * QKV_OK + 0], q_tile_tids[qt0 * QKV_OK + 1],
                    q_tile_tids[qt0 * QKV_OK + 2], q_tile_tids[qt0 * QKV_OK + 3],
                    q_tile_tids[qt0 * QKV_OK + 4], q_tile_tids[qt0 * QKV_OK + 5],
                    q_tile_tids[qt0 * QKV_OK + 6], q_tile_tids[qt0 * QKV_OK + 7],
                    q_tile_tids[qt0 * QKV_OK + 8], q_tile_tids[qt0 * QKV_OK + 9],
                    k_tile_tids[kt * QKV_OK + 0], k_tile_tids[kt * QKV_OK + 1],
                    k_tile_tids[kt * QKV_OK + 2], k_tile_tids[kt * QKV_OK + 3],
                    k_tile_tids[kt * QKV_OK + 4],
                    rms_tid,
                ],
            ) as qk_tid_h:
                q0 = h * Q_PER_KV * HEAD_DIM
                # Read q_proj[h] ONCE, scale by inv_rms -> q_chunk; feed both gamma + recip.
                q_slice = pl.row_expand_mul(
                    pl.slice(q_proj, [BATCH, Q_PER_KV * HEAD_DIM], [0, q0]), inv_rms_col
                )
                q_chunk = pl.reshape(q_slice, [BATCH * Q_PER_KV, HEAD_DIM])
                q_g = pl.col_expand_mul(q_chunk, q_norm_w)
                q_proj_norm = pl.assemble(
                    q_proj_norm, pl.reshape(q_g, [BATCH, Q_PER_KV * HEAD_DIM]), [0, q0]
                )
                q_ss = pl.row_sum(pl.mul(q_chunk, q_chunk))
                q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
                q_inv_states = pl.assemble(q_inv_states, q_inv, [h * BATCH * Q_PER_KV, 0])
                # K: same, read once.
                k0 = h * HEAD_DIM
                k_chunk = pl.row_expand_mul(pl.slice(k_proj, [BATCH, HEAD_DIM], [0, k0]), inv_rms_col)
                k_g = pl.col_expand_mul(k_chunk, k_norm_w)
                k_proj_norm = pl.assemble(k_proj_norm, k_g, [0, k0])
                k_ss = pl.row_sum(pl.mul(k_chunk, k_chunk))
                k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS)))
                k_inv_states = pl.assemble(k_inv_states, k_inv, [h * BATCH, 0])
            qk_tids[h] = qk_tid_h

        # rope GROUPED: NUM_ROPE_GROUPS grids of HEADS_PER_ROPE heads each, depping on
        # the group's qk + v tile (re-test @ swimlane level 2 to remove per-task overhead).
        # rope SINGLE spmd grid over ALL NUM_KV_HEADS*BATCH items — one launch (was 2
        # grouped grids = 2x per-grid scheduler overhead). Deps on all qk + v (rope follows).
        with pl.spmd(
            ROPE_CORES,
            name_hint="rope_qkv",
            deps=[
                qk_tids[0], qk_tids[1], qk_tids[2], qk_tids[3],
                qk_tids[4], qk_tids[5], qk_tids[6], qk_tids[7],
                rms_tid,
                v_tile_tids[0], v_tile_tids[1], v_tile_tids[2], v_tile_tids[3],
                v_tile_tids[4], v_tile_tids[5], v_tile_tids[6], v_tile_tids[7],
                v_tile_tids[8], v_tile_tids[9],
            ],
        ) as rope_tid:
            rope_core = pl.get_block_idx()
            for it in pl.pipeline(ROPE_GROUP_ITEMS_PER_CORE, stage=2):
                g_idx = rope_core * ROPE_ITEMS_PER_CORE + it
                ki = g_idx // BATCH
                b = g_idx % BATCH
                ctx_len = pl.read(seq_lens, [b])
                inv_rms_b = pl.read(inv_rms_states, [b, 0])
                pos = ctx_len - 1  # absolute position -> RoPE cos/sin row (NOT the cache row)
                # Paged write target for this row's current token: slot_mapping[b]
                # decomposes into (physical page, in-page offset). Same scheme prefill
                # uses; one physical page == one SEQ_TILE/BLOCK_SIZE band of the pool.
                wr_slot = pl.cast(pl.tensor.read(slot_mapping, [b]), pl.INDEX)
                wr_slot_block = wr_slot // BLOCK_SIZE
                wr_slot_offset = wr_slot - wr_slot_block * BLOCK_SIZE
                cos_lo = rope_cos[pos : pos + 1, 0:HALF_DIM]
                cos_hi = rope_cos[pos : pos + 1, HALF_DIM:HEAD_DIM]
                sin_lo = rope_sin[pos : pos + 1, 0:HALF_DIM]
                sin_hi = rope_sin[pos : pos + 1, HALF_DIM:HEAD_DIM]

                kv_col = ki * HEAD_DIM
                # K carries qk_norm gamma (qk_gamma); fold the deferred per-head qk_inv
                # scalar here. inv_rms cancels inside qk_norm, so no inv_rms factor on K.
                k_inv_b = pl.read(k_inv_states, [ki * BATCH + b, 0])
                k_full = pl.mul(k_proj_norm[b : b + 1, kv_col : kv_col + HEAD_DIM], k_inv_b)
                k_lo = k_full[:, 0:HALF_DIM]
                k_hi = k_full[:, HALF_DIM:HEAD_DIM]
                rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                cache_row = layer_cache_base + (wr_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + wr_slot_offset
                k_cache = pl.assemble(k_cache, k_cache[cache_row : cache_row + 1, :], [cache_row, 0])
                v_cache = pl.assemble(v_cache, v_cache[cache_row : cache_row + 1, :], [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_row, HALF_DIM])
                v_row_bf16 = pl.cast(
                    pl.mul(v_proj[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM], inv_rms_b),
                    target_type=pl.BF16,
                )
                v_cache = pl.assemble(v_cache, v_row_bf16, [cache_row, 0])

                q_base = ki * Q_PER_KV
                q_pad_row0 = b * NUM_KV_HEADS * Q_HEAD_PAD + ki * Q_HEAD_PAD
                q_inv_base = ki * BATCH * Q_PER_KV + b * Q_PER_KV
                for qj in pl.range(Q_PER_KV):
                    q_inv_bj = pl.read(q_inv_states, [q_inv_base + qj, 0])
                    q_head = pl.mul(
                        q_proj_norm[
                            b : b + 1, (q_base + qj) * HEAD_DIM : (q_base + qj + 1) * HEAD_DIM
                        ],
                        q_inv_bj,
                    )
                    q_lo = q_head[:, 0:HALF_DIM]
                    q_hi = q_head[:, HALF_DIM:HEAD_DIM]
                    q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                    q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(q_rot_lo, target_type=pl.BF16), [q_pad_row0 + qj, 0]
                    )
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(q_rot_hi, target_type=pl.BF16), [q_pad_row0 + qj, HALF_DIM]
                    )
                q_pad_zero = pl.cast(
                    pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),
                    target_type=pl.BF16,
                )
                all_q_padded = pl.assemble(all_q_padded, q_pad_zero, [q_pad_row0 + Q_HEAD_BATCH, 0])
        rope_grp_tids[0] = rope_tid

        # ── Scope 3b MLP-accumulator seeds, HOISTED between rope and attn. ──
        # gate/up/down accumulators are zeroed for the later split-K atomic-add
        # tasks; the seeds have NO data dependency on rope or attention, so placing
        # them here lets the scheduler overlap the (vector) zero-fills with the
        # fa_fused cube/vector work instead of serializing them at the head of the
        # MLP manual_scope. Their TASK_IDs (seed_tid / gate_seed_tid / up_seed_tid)
        # cross into the manual_scope below as explicit deps — the same pattern as
        # online_softmax's captured attn_done_tid. The accumulator create_tensor
        # calls move up with them (a seed must follow its tensor's allocation).
        down_acc_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        gate_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)
        up_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as seed_tid:
            for nb in pl.pipeline(DOWN_ON, stage=2):
                n0 = nb * DOWN_TN
                zero = pl.full([BATCH, DOWN_TN], dtype=pl.FP32, value=0.0)
                down_acc_all = pl.assemble(down_acc_all, zero, [0, n0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as gate_seed_tid:
            for nb in pl.pipeline(MLP_ON, stage=2):
                n0 = nb * MLP_TN
                zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                gate_acc_all = pl.assemble(gate_acc_all, zero, [0, n0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as up_seed_tid:
            for nb in pl.pipeline(MLP_ON, stage=2):
                n0 = nb * MLP_TN
                zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                up_acc_all = pl.assemble(up_acc_all, zero, [0, n0])

        # fa_fused: ONE mixed cube+vec root (QK -> softmax -> SV), BLOCK-LEVEL dense
        # static dispatch. Each grid-stride step processes exactly ONE real seq-block
        # (×GP_SIZE heads), decoded from the dense work table. Because the table holds
        # only real blocks, core = w % NUM_CORES distributes total_real_blocks evenly
        # across cores (≈1.25x of ideal), independent of per-batch length skew.
        with pl.spmd(
            NUM_CORES,
            name_hint="fa_fused",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
            deps=[work_tid, rope_grp_tids[0]],  # single rope grid now
        ) as fa_tid:
            fa_core = pl.get_block_idx()
            # Read the device-computed block count ON-DEVICE (inside the kernel) so
            # `fa_total` enters fa_fused as an INPUT and the normal task auto-dep
            # orders it after the `fa_work_build` producer. Reading it at
            # orchestration scope instead lowers to a host get_tensor_data that does
            # NOT wait for the producer (stale ~0 read → empty dispatch; see
            # KNOWN_ISSUES). This mirrors the existing on-device fa_work_table read.
            fa_total_blocks = pl.cast(pl.read(fa_total, [0, 0]), target_type=pl.INDEX)
            # Grid-stride over the dense real-block list: core fa_core owns table
            # entries fa_core, fa_core+NUM_CORES, … < fa_total_blocks.
            for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):
                fa_enc = pl.cast(pl.read(fa_work_table, [fa_w, 0]), target_type=pl.INDEX)
                fa_b = fa_enc // MAX_CTX_BLOCKS
                fa_p = fa_enc % MAX_CTX_BLOCKS
                fa_hg = 0  # HEAD_GROUPS == 1 (GP_SIZE == NUM_KV_HEADS)
                fa_ctx_len = pl.read(seq_lens, [fa_b])
                # Table holds only real blocks → exactly one block per entry, at fa_p.
                sb = fa_p  # logical KV block index (no inner loop — old p_blocks was 1)
                s0 = sb * SEQ_TILE
                valid_len = pl.min(SEQ_TILE, fa_ctx_len - s0)
                # Paged read: map logical block sb -> physical page via this request's
                # block_table row. SEQ_TILE == page_size, so one page is exactly one
                # contiguous SEQ_TILE-row slice of the pool (shared by all GP_SIZE heads
                # of this (b, block) work item below).
                fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)

                # Declarative software pipeline over the per-head gp loop: instead of
                # manually unrolling and reordering (QK0,QK1 -> softmax0,softmax1 ->
                # SV0,SV1), let pl.pipeline stage=2 overlap iteration i+1's QK (AIC)
                # with iteration i's softmax (AIV) through the C2V/V2C pipe.
                for gp in pl.pipeline(GP_SIZE, stage=2):
                    gi = fa_hg * GP_SIZE + gp
                    kvh = gi  # Q_GROUPS=1
                    q_pad_row_g = fa_b * NUM_KV_HEADS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                    q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]
                    g_base = (fa_b * NUM_KV_HEADS + gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD
                    cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE

                    # QK matmul (cube) -> C2V boundary move (first vec consumer).
                    k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]
                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                    scores_scaled = pl.mul(raw_scores, ATTN_SCALE)
                    # Mark only the Q_HEAD_BATCH (5) real rows valid (tile stays 16):
                    # the vec softmax and the cur_mi / cur_li GM stores then touch 5
                    # rows, not the padded 16. NOTE: this cannot shrink the AIC<->AIV
                    # C2V/V2C transfer — tpush/tpop always carry the full 16-row tile
                    # box (matmul M fractal).
                    scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_BATCH, valid_len)
                    scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                    cur_mi = pl.row_max(scores)
                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                    cur_li = pl.row_sum(exp_scores)
                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)

                    # SV matmul (cube) reads exp directly -> V2C boundary move.
                    v_tile = v_cache[cache_row : cache_row + SEQ_TILE, :]
                    oi_tmp = pl.matmul(exp_scores_bf16, v_tile, out_dtype=pl.FP32)
                    # The SV-matmul output spans the full padded 16 rows but only
                    # the first Q_HEAD_BATCH are real — mark 5 valid (tile stays 16)
                    # so the (dominant) all_oi_tmp GM write stores 5 rows instead of 16.
                    oi_tmp = pl.set_validshape(oi_tmp, Q_HEAD_BATCH, HEAD_DIM)

                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [g_base + sb * Q_HEAD_PAD, 0])
                    all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + sb * Q_HEAD_PAD, 0])
                    all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + sb * Q_HEAD_PAD, 0])

        # online_softmax: flat top-level spmd, writes attn_out directly. Reduces
        # per-block partials across ALL blocks of a lane (hence across the KV
        # partitions that produced them); auto-dep on the scratch serializes it
        # after every contributing fa_fused partition. Captured as `attn_done_tid`
        # (the dispatch's producer TaskId via the `with pl.spmd(...) as tid` form)
        # so the manual_scope out_proj tasks can take it as an explicit `deps=`
        # edge — auto-dep (tensormap) is suppressed inside manual_scope, so the
        # cross-boundary attn->out_proj order needs the TaskId. (This previously
        # required a dummy `attn_fence` pl.at task that read attn_out only to mint
        # the TaskId; capturing the spmd dispatch tid directly removes it.)
        with pl.spmd(
            NUM_CORES * 2, name_hint="online_softmax", deps=[fa_tid]
        ) as attn_done_tid:
            os_core = pl.get_block_idx()
            # Grid-stride over online_softmax work items (one per (b, kvh) lane).
            for os_spmd_idx in pl.range(os_core, OS_WORK, NUM_CORES * 2):
                os_b = os_spmd_idx // NUM_KV_HEADS
                os_gi = os_spmd_idx % NUM_KV_HEADS
                os_ctx_len = pl.read(seq_lens, [os_b])
                os_ctx_blocks = (os_ctx_len + SEQ_TILE - 1) // SEQ_TILE
                os_kvh = os_gi  # Q_GROUPS=1
                os_q_base = os_kvh * Q_PER_KV
                os_g_base = (os_b * NUM_KV_HEADS + os_gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD

                # Loop-carried accumulators: full Q_HEAD_PAD (16) rows. They can't be
                # valid-shaped because the recurrence (add / maximum) drops valid_shape,
                # which would break loop-carry type consistency. Only the per-block GM
                # reads below are trimmed to the 5 real rows via pl.slice valid_shape.
                oi = all_oi_tmp[os_g_base : os_g_base + Q_HEAD_PAD, :]
                mi = all_cur_mi[os_g_base : os_g_base + Q_HEAD_PAD, :]
                li = all_cur_li[os_g_base : os_g_base + Q_HEAD_PAD, :]
                for sb in pl.pipeline(1, os_ctx_blocks, stage=2):
                    rec = os_g_base + sb * Q_HEAD_PAD
                    # Partial load: tile stays Q_HEAD_PAD (16) rows, GM transfer = the 5
                    # real rows (valid_shape on the slice → tile.load valid_shapes).
                    oi_tmp_valid = pl.slice(
                        all_oi_tmp, [Q_HEAD_PAD, HEAD_DIM], [rec, 0], valid_shape=[Q_HEAD_BATCH, HEAD_DIM]
                    )
                    online_cur_mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [rec, 0], valid_shape=[Q_HEAD_BATCH, 1])
                    online_cur_li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [rec, 0], valid_shape=[Q_HEAD_BATCH, 1])
                    mi_new = pl.maximum(mi, online_cur_mi)
                    alpha = pl.exp(pl.sub(mi, mi_new))
                    beta = pl.exp(pl.sub(online_cur_mi, mi_new))
                    li = pl.add(pl.mul(alpha, li), pl.mul(beta, online_cur_li))
                    oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp_valid, beta))
                    mi = mi_new

                ctx = pl.row_expand_div(oi, li)
                ctx_valid = ctx[0:Q_HEAD_BATCH, :]
                ctx_flat_bf16 = pl.cast(
                    pl.reshape(ctx_valid, [1, Q_HEAD_BATCH * HEAD_DIM]), target_type=pl.BF16
                )
                attn_out = pl.assemble(attn_out, ctx_flat_bf16, [os_b, os_q_base * HEAD_DIM])

        # Scope-3 allocations. (down_acc_all / gate_acc_all / up_acc_all are created
        # earlier, alongside their hoisted seed tasks between rope and attn.)
        attn_proj_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        post_norm_partial = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)  # raw residual h1 (add-back); FP32 (was BF16)
        mlp_norm_in = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # h1 * post_gamma (gate/up input)
        inv_rms_tile = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
        mlp_tile = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)

        # Out_seed zeros attn_proj_fp32 in OUT_TN-wide chunks so out_proj
        # split-K tasks can atomic-add into it.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as out_seed_tid:
            for nb in pl.pipeline(N_SPLITS_OUT, stage=2):
                out_seed_n0 = nb * OUT_TN
                out_zero = pl.full([BATCH, OUT_TN], dtype=pl.FP32, value=0.0)
                attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_zero, [0, out_seed_n0])

        # ── Scope 3b: manual_scope MLP block. ──
        silu_tids = pl.array.create(MLP_ON, pl.TASK_ID)
        # down_tids is hoisted to orchestration scope (declared before this
        # manual_scope) so the post-scope consolidated writer can gate on it;
        # it is FILLED here in the down_proj loop below.
        gate_tids = pl.array.create(MLP_ON * K_SPLITS_MLP, pl.TASK_ID)
        up_tids = pl.array.create(MLP_ON * K_SPLITS_MLP, pl.TASK_ID)
        cast_tids = pl.array.create(K_SPLITS_MLP, pl.TASK_ID)
        out_tids = pl.array.create(N_SPLITS_OUT * K_SPLITS_OUT, pl.TASK_ID)

        # Split-K split-N out_proj: 10 × 5 = 50 atomic-add tasks.
        for n_out_proj in pl.parallel(N_SPLITS_OUT):
            n_op = n_out_proj * OUT_TN
            for k_split_out in pl.range(K_SPLITS_OUT):
                k_op = k_split_out * OUT_TK
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="out_proj",
                    deps=[out_seed_tid, attn_done_tid],
                ) as out_tid:
                    out_a0 = attn_out[:, k_op : k_op + OUT_INNER_TK]
                    out_w0 = wo[layer_hidden_base + k_op : layer_hidden_base + k_op + OUT_INNER_TK, n_op : n_op + OUT_TN]
                    out_c_acc = pl.matmul(out_a0, out_w0, out_dtype=pl.FP32)
                    for out_lk in pl.pipeline(1, OUT_N_SUB_K, stage=2):
                        out_ks_off = out_lk * OUT_INNER_TK
                        out_a_k = attn_out[:, k_op + out_ks_off : k_op + out_ks_off + OUT_INNER_TK]
                        out_w_k = wo[
                            layer_hidden_base + k_op + out_ks_off : layer_hidden_base + k_op + out_ks_off + OUT_INNER_TK,
                            n_op : n_op + OUT_TN,
                        ]
                        out_c_acc = pl.matmul_acc(out_c_acc, out_a_k, out_w_k)
                    attn_proj_fp32 = pl.assemble(
                        attn_proj_fp32, out_c_acc, [0, n_op], atomic=pl.AtomicType.Add
                    )
                out_tids[n_out_proj * K_SPLITS_OUT + k_split_out] = out_tid

        # Tiled residual + BF16 cast.
        for k_slice in pl.unroll(K_SPLITS_MLP):
            k_base = k_slice * MLP_K_SLICE
            n_split_base = k_slice * N_PER_CAST_K
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="residual_rms_cast",
                deps=[
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 0],
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 1],
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 2],
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 3],
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 4],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 0],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 1],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 2],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 3],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 4],
                ],
            ) as cast_tid_k:
                for kb in pl.pipeline(MLP_K_SLICE // K_CHUNK, stage=2):
                    k0 = k_base + kb * K_CHUNK
                    attn_chunk = attn_proj_fp32[:, k0 : k0 + K_CHUNK]
                    hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]  # FP32 already
                    resid_fp32 = pl.add(attn_chunk, hidden_chunk)
                    # Raw residual h1 — added back after down_proj (must NOT be gamma-scaled).
                    post_norm_partial = pl.assemble(post_norm_partial, resid_fp32, [0, k0])  # FP32 (no BF16 cast)
                    # Explicit post-RMS gamma: gate/up input = h1 * post_gamma. gamma is
                    # per-K (the matmul contraction dim) so it canNOT defer past the matmul
                    # like inv_rms does — it scales the input here (with raw w_gate/w_up).
                    post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                    mlp_norm_in = pl.assemble(
                        mlp_norm_in,
                        pl.cast(pl.col_expand_mul(resid_fp32, post_gamma), target_type=pl.BF16),
                        [0, k0],
                    )
            cast_tids[k_slice] = cast_tid_k

        # RMS reduction reads all of attn_proj_fp32.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rms_reduce", deps=[out_tids]) as reduce_tid:
            sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=2):
                k0 = kb * K_CHUNK
                attn_chunk = attn_proj_fp32[:, k0 : k0 + K_CHUNK]
                hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]  # FP32 already
                resid_chunk = pl.add(attn_chunk, hidden_chunk)
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH]),
                )
            post_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
            post_inv_rms_col = pl.reshape(post_inv_rms, [BATCH, 1])
            inv_rms_tile = pl.assemble(inv_rms_tile, post_inv_rms_col, [0, 0])

        # Split-K gate + up interleaved.
        for n_out in pl.parallel(MLP_ON):
            n0 = n_out * MLP_TN
            for k_split in pl.range(K_SPLITS_MLP):
                k0 = k_split * MLP_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="gate_proj",
                    deps=[cast_tids[k_split], gate_seed_tid],
                ) as gate_tid:
                    a0 = mlp_norm_in[:, k0 : k0 + MLP_INNER_TK]
                    w0 = w_gate[layer_hidden_base + k0 : layer_hidden_base + k0 + MLP_INNER_TK, n0 : n0 + MLP_TN]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):
                        ks_off = lk * MLP_INNER_TK
                        a_k = mlp_norm_in[:, k0 + ks_off : k0 + ks_off + MLP_INNER_TK]
                        w_k = w_gate[layer_hidden_base + k0 + ks_off : layer_hidden_base + k0 + ks_off + MLP_INNER_TK, n0 : n0 + MLP_TN]
                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)
                    gate_acc_all = pl.assemble(gate_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)
                gate_tids[n_out * K_SPLITS_MLP + k_split] = gate_tid

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="up_proj",
                    deps=[cast_tids[k_split], up_seed_tid],
                ) as up_tid:
                    a0 = mlp_norm_in[:, k0 : k0 + MLP_INNER_TK]
                    w0 = w_up[layer_hidden_base + k0 : layer_hidden_base + k0 + MLP_INNER_TK, n0 : n0 + MLP_TN]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):
                        ks_off = lk * MLP_INNER_TK
                        a_k = mlp_norm_in[:, k0 + ks_off : k0 + ks_off + MLP_INNER_TK]
                        w_k = w_up[layer_hidden_base + k0 + ks_off : layer_hidden_base + k0 + ks_off + MLP_INNER_TK, n0 : n0 + MLP_TN]
                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)
                    up_acc_all = pl.assemble(up_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)
                up_tids[n_out * K_SPLITS_MLP + k_split] = up_tid

        # silu.
        for n_out in pl.parallel(MLP_ON):
            n0 = n_out * MLP_TN
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="silu",
                deps=[
                    reduce_tid,
                    gate_tids[n_out * K_SPLITS_MLP + 0],
                    gate_tids[n_out * K_SPLITS_MLP + 1],
                    gate_tids[n_out * K_SPLITS_MLP + 2],
                    gate_tids[n_out * K_SPLITS_MLP + 3],
                    gate_tids[n_out * K_SPLITS_MLP + 4],
                    up_tids[n_out * K_SPLITS_MLP + 0],
                    up_tids[n_out * K_SPLITS_MLP + 1],
                    up_tids[n_out * K_SPLITS_MLP + 2],
                    up_tids[n_out * K_SPLITS_MLP + 3],
                    up_tids[n_out * K_SPLITS_MLP + 4],
                ],
            ) as silu_tid:
                inv_rms_chunk = inv_rms_tile[:, 0:1]
                for sub in pl.pipeline(SILU_INNER_CHUNKS, stage=2):
                    silu_off = n0 + sub * MLP_OUT_CHUNK
                    gate_chunk = gate_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]
                    up_chunk = up_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]
                    scaled_gate = pl.row_expand_mul(gate_chunk, inv_rms_chunk)
                    scaled_up = pl.row_expand_mul(up_chunk, inv_rms_chunk)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(scaled_gate)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(scaled_gate, sigmoid), scaled_up)
                    mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, silu_off])
            silu_tids[n_out] = silu_tid

        for n_out in pl.parallel(DOWN_ON):
            n0 = n_out * DOWN_TN
            for k_split in pl.range(K_SPLITS):
                k0 = k_split * DOWN_TN
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="down_proj",
                    deps=[seed_tid, silu_tids[k_split]],
                ) as down_tid:
                    a0 = mlp_tile[:, k0 : k0 + DOWN_TK]
                    w0 = w_down[layer_inter_base + k0 : layer_inter_base + k0 + DOWN_TK, n0 : n0 + DOWN_TN]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, N_SUB_K, stage=2):
                        ks_off = lk * DOWN_TK
                        a_k = mlp_tile[:, k0 + ks_off : k0 + ks_off + DOWN_TK]
                        w_k = w_down[layer_inter_base + k0 + ks_off : layer_inter_base + k0 + ks_off + DOWN_TK, n0 : n0 + DOWN_TN]
                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)
                    down_acc_all = pl.assemble(down_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)
                down_tids[n_out * K_SPLITS + k_split] = down_tid

    # ── down_cast_residual (DOWN_ON-way, OUTSIDE manual_scope): the residual
    # add (down_acc_all + post_norm_partial, both FP32) is the layer output,
    # emitted as DOWN_ON sliced writers of `out` in the AUTO-DEP region —
    # restoring the baseline's 5-way parallelism while keeping the fusion
    # (no scratch out_partial, no out_consolidate copy).
    #
    # Why outside manual_scope: inside manual_scope, auto-dep (tensormap)
    # registration is suppressed (explicit deps= only), so partial writers
    # register NO tensormap edge to downstream caller readers (proven on
    # device: next_hidden's writers had no edge to copy_out → garbage). In the
    # auto-dep region each sliced writer registers `out`; OptimizeOrchTensors
    # Pattern-5 narrows the footprint to a window, and the runtime tensormap
    # check is region-precise per-dim, so the 5 writers do not serialize
    # against each other while the next layer's x_gamma still waits on all 5.
    #
    # Deps: each block needs only its own K_SPLITS (=17) down_proj atomic-add
    # tids (per-index list comprehension), so block n starts as soon as its
    # column slab is accumulated. Transitivity covers post_norm_partial: every
    # down_proj K-slice deps on a silu task, which deps on ALL cast_tids =
    # residual_rms_cast — the full producer of post_norm_partial.
    # dcr_xgamma as a SINGLE pl.spmd(DOWN_ON) dispatch (was DOWN_ON separate pl.parallel
    # pl.at tasks). PERF: the separate-task form WAW-serialized the DOWN_ON sliced writers
    # of `out` / `normed_out` on this runtime — the OptimizeOrchTensors region-narrowing
    # that was meant to keep them parallel did NOT fire (measured: the 5 dcr ran SERIAL on
    # one core, ~35us, both at the chunk tail AND every layer boundary). A single spmd
    # dispatch's blocks are inherently parallel (exactly like x_gamma's disjoint
    # normed_states writes), so the disjoint-slice writes run on DOWN_ON cores. Trade-offs:
    # the dispatch deps on ALL down_proj tids (not per-column), but down_proj finishes
    # ~together; and the carry is ONE dispatch tid for all DOWN_ON slabs (the next layer's
    # rms_recip/QKV/seeds wait on the whole dcr dispatch — fine once it is ~3us not ~35us).
    with pl.spmd(
        DOWN_ON,
        name_hint="dcr_xgamma",
        deps=[down_tids[i] for i in range(DOWN_ON * K_SPLITS)],
    ) as dcr_tid:
        n_out = pl.tile.get_block_idx()
        n0 = n_out * DOWN_TN
        # OUTPUT 1: layer residual (down_acc + post_norm, both FP32) -> `out` (cur).
        out_chunk = pl.add(
            down_acc_all[:, n0 : n0 + DOWN_TN], post_norm_partial[:, n0 : n0 + DOWN_TN]
        )
        out = pl.assemble(out, out_chunk, [0, n0])
        # OUTPUT 2: NEXT layer's x*gamma from the same in-register FP32 chunk (no GM
        # re-read of `out`). gamma row clamped via next_gamma_idx (last layer unused).
        gamma_next = pl.slice(input_rms_weight, [1, DOWN_TN], [next_gamma_idx, n0])
        xg = pl.col_expand_mul(out_chunk, gamma_next)
        normed_out = pl.assemble(normed_out, pl.cast(xg, target_type=pl.BF16), [0, n0])
    # One spmd dispatch tid carries BOTH out + normed_out for all DOWN_ON slabs (the
    # per-block tids of the old pl.parallel form are gone): the next layer's rms_recip
    # / seeds (read prev_out_tids[*]) and QKV (read prev_normed_tids[*]) all wait on
    # the single dcr dispatch. Fill every slab slot so the DOWN_ON-wide carry (sized
    # for copy_hidden's layer-0 seed) stays valid.
    for _slab in pl.unroll(DOWN_ON):
        prev_out_tids[_slab] = dcr_tid
        prev_normed_tids[_slab] = dcr_tid
    return out

_CHUNK_NLAYERS = 1

@pl.jit
def decode_fwd_layers(  # noqa: PLR0913 — fused decode of a CONTIGUOUS layer CHUNK, no LM head
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    # Fused decode of _CHUNK_NLAYERS consecutive layers, output = hidden (NO LM head).
    # Used to run all 40 layers in a few dispatches instead of one — a single 40-layer
    # dispatch exceeds the device AICPU stream-sync timeout (PLATFORM_STREAM_SYNC_TIMEOUT
    # _MS=2000ms). Each layer's output is made visible to the next by _decode_layer's
    # out_consolidate (the fused-decode dependency fix). The caller passes the weight /
    # KV-cache SLICES for the chunk's layers (stacked [_CHUNK_NLAYERS*dim, ...]); the body
    # indexes them with layer_idx 0.._CHUNK_NLAYERS-1.
    # FP32 inter-layer carry (matches _decode_layer's FP32-carry signature). The
    # chunk input/output hidden are BF16 (host passes BF16 between chunks), so cast
    # BF16->FP32 at the chunk head (copy_hidden) and FP32->BF16 at the tail (copy_out),
    # mirroring decode_fwd's embed-in / lm-head-in casts. (Without these the BF16 cur
    # hits _decode_layer's FP32 x_gamma/rms_recip -> ptoas bfloat16 type error.)
    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    carry_tids = pl.array.create(DOWN_ON, pl.TASK_ID)
    for cb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden") as ch_tid:
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                cur = pl.assemble(
                    cur,
                    pl.cast(
                        pl.slice(hidden_states, [BATCH, RMSNORM_K_CHUNK], [cb0, ck0]),
                        target_type=pl.FP32,
                    ),
                    [cb0, ck0],
                )
        for cseed in pl.range(DOWN_ON):
            carry_tids[cseed] = ch_tid

    # Pre-loop x_gamma_0: chunk layer 0's normed = cur * gamma_0 (mirrors decode_fwd).
    normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    carry_normed_tids = pl.array.create(DOWN_ON, pl.TASK_ID)
    with pl.manual_scope():
        for xg_n in pl.range(XG_BLOCKS):
            xg_k0 = xg_n * (HIDDEN // XG_BLOCKS)
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="x_gamma0", deps=[carry_tids[xg_n]]
            ) as xg0_tid:
                for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK // XG_BLOCKS, stage=2):
                    k0 = xg_k0 + kb * RMSNORM_K_CHUNK
                    x_chunk = cur[:, k0 : k0 + RMSNORM_K_CHUNK]
                    gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [0, k0])
                    xg = pl.col_expand_mul(x_chunk, gamma)
                    normed = pl.assemble(normed, pl.cast(xg, target_type=pl.BF16), [0, k0])
            carry_normed_tids[xg_n] = xg0_tid

    for i in pl.range(_CHUNK_NLAYERS):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        next_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        next_gamma_idx = pl.min(i + 1, _CHUNK_NLAYERS - 1)
        cur = _decode_layer(
            cur, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache, wo, w_gate, w_up, w_down,
            post_rms_weight, next_hidden, normed, next_normed, i, next_gamma_idx,
            carry_tids, carry_normed_tids,
        )
        normed = next_normed
    for ob0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_out"):
            for okb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ok0 = okb * RMSNORM_K_CHUNK
                out = pl.assemble(
                    out,
                    pl.cast(
                        pl.slice(cur, [BATCH, RMSNORM_K_CHUNK], [ob0, ok0]), target_type=pl.BF16
                    ),
                    [ob0, ok0],
                )
    return out
