import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ
T_DYN = pl.dynamic('T_DYN')
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)
from config import D
from config import HC_MULT
from config import HC_DIM
EPS = M.rms_norm_eps
HC_EPS = M.hc_eps
HC_DIM_INV = 1.0 / HC_DIM
HC_PAD = 16
T_TILE = 8
LINEAR_T_TILE = 16
TRANSPOSE_T_TILE = 8
RMS_K_CHUNK = 512
LINEAR_K_CHUNK = 256
D_CHUNK = 512
LINEAR_OK = 8
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
D_BLOCKS = D // D_CHUNK
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // LINEAR_K_CHUNK

@pl.jit.inline
def hc_head(x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32], hc_head_scale: pl.Tensor[[1], pl.FP32], hc_head_base: pl.Tensor[[HC_MULT], pl.FP32], y: pl.Tensor[[T_DYN, D], pl.BF16]):
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = pl.max(t_dim, LINEAR_T_TILE)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, D])
    inv_rms = pl.create_tensor([T_MAX, 1], dtype=pl.FP32)
    mixes_raw = pl.create_tensor([T_MAX, HC_PAD], dtype=pl.FP32)
    pre_t = pl.create_tensor([HC_PAD, T_MAX], dtype=pl.FP32)
    for t in pl.spmd(t_dim // T_TILE, name_hint='hc_head_rms', allow_early_resolve=True):
        t0 = t * T_TILE
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(RMS_K_BLOCKS, stage=4):
            k0 = kb * RMS_K_CHUNK
            x_chunk = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]))
        head_var = pl.add(pl.mul(sq_sum, HC_DIM_INV), EPS)
        inv = pl.reshape(pl.rsqrt(head_var, high_precision=True), [T_TILE, 1])
        inv_rms = pl.assemble(inv_rms, inv, [t0, 0])
    for tc in pl.spmd(t_linear // T_TILE, name_hint='hc_head_seed', allow_early_resolve=True):
        ts0 = tc * T_TILE
        mixes_raw[ts0:ts0 + T_TILE, 0:HC_PAD] = pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=0.0)
    for task in pl.spmd(t_linear // LINEAR_T_TILE * LINEAR_OK, name_hint='hc_head_linear', allow_early_resolve=True):
        t0 = task // LINEAR_OK * LINEAR_T_TILE
        k_base = task % LINEAR_OK * LINEAR_K_PER_SPLIT
        t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        acc = pl.create_tensor([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
            k0 = k_base + kb * LINEAR_K_CHUNK
            x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, k0], valid_shape=[t_rows, LINEAR_K_CHUNK])
            w_chunk = pl.slice(hc_head_fn, [HC_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[HC_MULT, LINEAR_K_CHUNK])
            if kb == 0:
                acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
        mixes_raw = pl.assemble(mixes_raw, acc, [t0, 0], atomic=pl.AtomicType.Add)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='hc_head_pre_fused', allow_early_resolve=True):
        scale = pl.read(hc_head_scale, [0])
        base = pl.reshape(pl.slice(hc_head_base, [HC_PAD], [0], valid_shape=[HC_MULT]), [1, HC_PAD])
        for t0 in pl.pipeline(0, t_dim, TRANSPOSE_T_TILE, stage=2):
            scaled = pl.row_expand_mul(mixes_raw[t0:t0 + TRANSPOSE_T_TILE, 0:HC_PAD], inv_rms[t0:t0 + TRANSPOSE_T_TILE, 0:1])
            logits = pl.add(pl.mul(scaled, scale), pl.col_expand(scaled, base))
            pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
            pre_t[:, t0:t0 + TRANSPOSE_T_TILE] = pl.transpose(pre_val, axis1=0, axis2=1)
    for t0 in pl.parallel(0, t_dim, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='hc_head_reduce', allow_early_resolve=True):
            pre0 = pl.reshape(pre_t[0:1, t0:t0 + T_TILE], [T_TILE, 1])
            pre1 = pl.reshape(pre_t[1:2, t0:t0 + T_TILE], [T_TILE, 1])
            pre2 = pl.reshape(pre_t[2:3, t0:t0 + T_TILE], [T_TILE, 1])
            pre3 = pl.reshape(pre_t[3:4, t0:t0 + T_TILE], [T_TILE, 1])
            for db in pl.pipeline(D_BLOCKS, stage=2):
                d0 = db * D_CHUNK
                x_h0 = x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_CHUNK]
                x_h1 = x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_CHUNK]
                x_h2 = x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_CHUNK]
                x_h3 = x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_CHUNK]
                y_tile = pl.add(pl.add(pl.row_expand_mul(x_h0, pre0), pl.row_expand_mul(x_h1, pre1)), pl.add(pl.row_expand_mul(x_h2, pre2), pl.row_expand_mul(x_h3, pre3)))
                y_flat[t0:t0 + T_TILE, d0:d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode='rint')
    y = pl.reshape(y_flat, [t_dim, D])
    return y
