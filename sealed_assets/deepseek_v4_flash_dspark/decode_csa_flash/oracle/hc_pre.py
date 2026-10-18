# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-dep-gen  # CI marker: full-occupancy pl.system.syncall -> dep_gen (DFX) trips 507018 (pypto#1931)
"""DeepSeek-V4 hc_pre -- hyper-connection pre-mix, fused into ONE syncall-barriered task.

x[T, hc, D] holds hc streams of the hidden state (x_flat = x.reshape(T, hc*D)). With the
projection hc_fn[mix_hc, hc*D], the per-group scales hc_scale[3] and biases hc_base[mix_hc]:

    inv_rms[T, 1]   = rsqrt(mean_k(x_flat^2) + eps)
    mixes[T, mix_hc]= inv_rms * (x_flat @ hc_fn^T)              # 1/rms deferred past the matmul
    pre [T, hc]     = sigmoid(mixes[:, :hc]   * s0 + base) + eps
    post[T, hc]     = 2 * sigmoid(mixes[:, hc:2hc] * s1 + base)
    comb[T, hc, hc] = sinkhorn(softmax(reshape(mixes[:, 2hc:] * s2 + base, hc, hc)))
    x_mixed[T, D]   = sum_h pre[:, h] * x[:, h, :]

TWO IMPLEMENTATIONS, selected by the ``HC_PRE_IMPL`` module flag (env ``DSV4_HC_PRE_IMPL``
or ``--impl``); both are UNIFIED over decode + prefill and compute the identical math above:
  * ``_hc_pre_syncall`` (default) -- the #684 fusion documented below: ONE full-occupancy
    ``pl.spmd(24)`` with 2 hard ``pl.system.syncall`` barriers, run with dep_gen OFF.
  * ``_hc_pre_separate`` -- the pre-#684 structure: each work-type is its OWN ``pl.spmd``
    task (cast / rms / seed / linear / split_pre_post / write_post / comb_sinkhorn / mix_x),
    ordered by the runtime task graph (dep_gen ON), tile sizes aligned to the syncall path.
The rest of this docstring describes the default ``_hc_pre_syncall`` fusion.

FUSION (unified decode + prefill): the whole op is ONE ``pl.spmd(NUM_CORES=24)`` launch --
24 persistent blocks == 24 AIC + 48 AIV == every 910B core (the full-occupancy contract a
hard ``pl.system.syncall(core_type="mix")`` requires). The body is 3 phases separated by 2
barriers; each barrier publishes the prior phase's cross-core HBM writes:

    Phase A  cast (x BF16 -> x_fp32 FP32)  +  seed (zero mixes_raw)          AIV
    ── syncall ──  (x_fp32 + zeroed mixes_raw visible)
    Phase B  linear (split-K matmul -> mixes_raw, AIC)  +  rms (split-K SoS -> sq_sum_acc, AIV)
    ── syncall ──  (sq_sum_acc + all LINEAR_OK atomic-add partials visible)
    Phase D  per task: rsqrt(sq_sum) + scale/sigmoid GATE, then                    AIV
             comb_sinkhorn (-> comb) | mix_x (-> x_mixed) | write_post (-> post)

The old Phase C (a separate rsqrt+scale+sigmoid gate pass, published by a 3rd barrier) is
FOLDED into Phase D: since the C->D handoff was cross-core (different grid-stride striping),
it needed a barrier -- but each D task can just recompute the ONE gate it consumes from the
barrier-2-published mixes_raw + sq_sum_acc on its OWN core (rsqrt+sigmoid is nearly free on
this latency-bound kernel). That deletes a whole hard FFTS barrier (-8% decode / -11% prefill
on the device L2 swimlane, latency-bound so barrier removal > byte removal). comb_logits /
post_pad_store survive only as SAME-CORE scratch (assemble then load-back, no barrier) because
their downstream pl.load->pl.store needs an HC_PAD-wide 32B-aligned tile; the pre gate is
consumed in tensor-world so it needs no buffer.

Scheduling within a phase: each independent task-type is its OWN unconditional grid-stride
loop (cube work strides over ``core`` 0..23, vector work over the global AIV-lane id
``core*2 + aiv_id`` 0..47). Grid-stride IS the static round-robin schedule -- a pool larger
than the core count wraps into further waves; a smaller one leaves extra cores running a
0-trip loop that falls straight through to the next type (so a lane with no sinkhorn work
starts mix_x immediately, overlapping the sinkhorn on a busy lane). LOOP ORDER is the
scheduling policy: Phase D runs comb_sinkhorn FIRST -- its 20-iteration serial recurrence is
a latency floor no core count shortens, so starting it first overlaps it with the
throughput-bound mix_x on the other lanes.

Split-K linear atomic-adds LINEAR_OK FP32 partials into mixes_raw; Barrier 1's zero-seed +
Barrier 2's publish replace the old seed->RMW WAW edge. assemble(atomic=Add) is device-only
and the hard barrier is not modeled by the a2a3sim / a5sim simulators, so those two sim CI
checks are skipped for hc_pre -- validate on real device. mix_hc(24) pads to a 32-wide cube
N (MIX_PAD) and hc(4) to an 8-wide vector row (HC_PAD).
"""


import os


from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ



# Implementation selector (both versions are UNIFIED -- one code path for decode AND
# prefill; no separate decode/prefill dispatch):
#   "syncall"  (default) -- the #684 fused kernel: ONE full-occupancy pl.spmd(24) with
#               2 hard pl.system.syncall barriers; runs with dep_gen OFF (pypto#1931).
#   "separate" -- the pre-#684 multi-scope structure: each work-type is its own pl.spmd
#               task, ordered by the runtime task graph (dep_gen ON), applied to ALL T.
# Switch via env DSV4_HC_PRE_IMPL={syncall,separate} or by reassigning this module global
# before the kernel is traced (the __main__ below wires it to --impl).
HC_PRE_IMPL = os.environ.get("DSV4_HC_PRE_IMPL", "separate").lower()


D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

MIX_PAD = 32  # mix_hc (24) padded to a 32-wide cube N / vector row
HC_PAD = 8  # hc (4) padded for 32B-aligned vector ops

# Full-occupancy launch: NUM_CORES persistent blocks == physical AIC count (910B = 24), so a
# hard mix-syncall reaches every AIC + AIV. Must equal the physical count exactly -- pypto does
# NOT cap the launch, and >24 makes the runtime multi-round the blocks so some physical core is
# in a later round when the barrier fires -> AICore timeout 507018.
NUM_CORES = 24

# tiling (unchanged)
T_TILE = 8  # vector row-tile (RMS / cast / split / mix_x)
LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile
COMB_T_TILE = 8  # sinkhorn row-tile
RMS_K_CHUNK = 512  # cast / rms K-fragment
LINEAR_K_CHUNK = 256  # cube K-fragment per matmul_acc (32x256x4 FP32 weight fits L0B)
D_CHUNK = 256  # mix_x inner D-fragment (BF16 load = 1KB, 512B-aligned)
D_SPMD = 1024  # mix_x D per spmd block: decode fans 4096 reduce over D/D_SPMD cores
CAST_K_SPMD = 2048  # cast K per spmd block: decode fans the BF16->FP32 cast over HC_DIM/CAST_K_SPMD cores
# Split the K=HC_DIM reduction into LINEAR_OK slices that atomic-add their FP32
# partials, filling idle cubes at small T (decode: 1 token-tile -> LINEAR_OK
# cube tasks) and shortening each task's matmul_acc chain. Higher OK fills more
# decode cubes; prefill (8 token-tiles) packs OK*8 tasks into waves of ~24.
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // LINEAR_K_CHUNK

# Split the RMS sum-of-squares K reduction over RMS_OK cores, mirroring LINEAR_OK: at decode
# (1 token-tile) the full 16384-wide reduce is otherwise a single-lane straggler. Each
# (token-tile, K-slice) task atomic-adds its FP32 partial sum-of-squares into sq_sum_acc
# (zero-seeded in Phase A); Phase C reads the barrier-published total and applies rsqrt inline
# per group (no separate inv_rms buffer / no within-phase RAW).
RMS_OK = 16
RMS_K_PER_SPLIT = HC_DIM // RMS_OK
RMS_CHUNKS_PER_SPLIT = RMS_K_PER_SPLIT // RMS_K_CHUNK

# per-phase fan-out factors (compile-time constants; the token-tile factor is dynamic in T)
CAST_KS = HC_DIM // CAST_K_SPMD  # cast tasks per token-tile (8)
MIXX_DS = D // D_SPMD  # mix_x tasks per token-tile (4)

assert HC_MULT == 4, (
    f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}; "
    "regenerate the pre0..pre3 / row0..row3 unrolling before using it."
)
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % LINEAR_T_TILE == 0 or DECODE_BATCH * DECODE_SEQ <= LINEAR_T_TILE
assert HC_DIM % LINEAR_K_CHUNK == 0
assert HC_DIM % CAST_K_SPMD == 0 and CAST_K_SPMD % RMS_K_CHUNK == 0
assert HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0
assert D % D_SPMD == 0 and D_SPMD % D_CHUNK == 0



def _golden_a2a3_cube_linear(x_flat_2d, hc_fn):
    """Emulate the A2/A3 f32 Cube MAD reduction used by the HC projection."""
    import torch

    cube_acc_k = 4
    group_block = 64
    assert LINEAR_K_PER_SPLIT % cube_acc_k == 0
    groups_per_split = LINEAR_K_PER_SPLIT // cube_acc_k

    t_dim = x_flat_2d.shape[0]
    x_groups = x_flat_2d.reshape(t_dim, 1, LINEAR_OK, groups_per_split, cube_acc_k)
    w_groups = hc_fn.reshape(1, MIX_HC, LINEAR_OK, groups_per_split, cube_acc_k)
    split_mixes = torch.zeros(t_dim, MIX_HC, LINEAR_OK, dtype=torch.float32)
    for group0 in range(0, groups_per_split, group_block):
        group_dots = (
            x_groups[..., group0:group0 + group_block, :].double()
            * w_groups[..., group0:group0 + group_block, :].double()
        ).sum(dim=-1)
        for group in range(group_dots.shape[-1]):
            split_mixes = (split_mixes.double() + group_dots[..., group]).float()

    # Atomic completion order is not a semantic guarantee.  Canonical split
    # order is the deterministic golden representative.  Other legal orders
    # differ by only a few FP32 ULPs; the output tolerances cover the rare case
    # where that difference crosses a BF16 boundary.
    mixes = torch.zeros(t_dim, MIX_HC, dtype=torch.float32)
    for split in range(LINEAR_OK):
        mixes += split_mixes[..., split]
    return mixes


def golden_hc_pre(tensors):
    """Torch reference matching the target HC-pre reduction and nonlinearities."""
    import torch

    x = tensors["x"].float()  # [T, hc, D]
    hc_fn = tensors["hc_fn"].float()  # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()  # [3]
    hc_base = tensors["hc_base"].float()  # [mix_hc]

    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)

    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)

    # A2/A3 f32 Cube MAD accumulates four consecutive products before rounding
    # the updated accumulator to FP32.  Reproduce that target-defined reduction
    # instead of using Torch's different reduction tree.
    mixes = _golden_a2a3_cube_linear(x_flat_2d, hc_fn)
    mixes *= rsqrt

    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(t_dim, HC_MULT, HC_MULT)

    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = (y0 + y1) + (y2 + y3)

    # Match the kernel's mode="rint" cast (round to nearest, ties to even).
    tensors["x_mixed"][:] = y.to(torch.bfloat16).reshape(t_dim, D)
    tensors["post"][:] = post_t.reshape(t_dim, HC_MULT)
    tensors["comb"][:] = comb_t.reshape(t_dim, HC_MULT * HC_MULT)


