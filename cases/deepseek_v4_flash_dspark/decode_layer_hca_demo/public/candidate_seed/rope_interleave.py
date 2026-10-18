"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_hca_demo."""

from config import CASE_CONFIG as M
from config import TP
from config import DECODE_BATCH
import pypto.language as pl

B_DYN = pl.dynamic('B_DYN')
B_MAX = DECODE_BATCH // TP
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
B_TILE = 4

@pl.jit.inline
def rope_interleave(cos_half: pl.Tensor[[B_DYN, HALF_ROPE], pl.FP32], sin_half: pl.Tensor[[B_DYN, HALF_ROPE], pl.FP32], cos_il: pl.Tensor[[B_MAX, ROPE_HEAD_DIM], pl.FP32], sin_signed: pl.Tensor[[B_MAX, ROPE_HEAD_DIM], pl.FP32]):
    """Expand half-width cos/sin rows to the interleaved, sign-folded rope layout."""
    b_dim = pl.tensor.dim(cos_half, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='rope_interleave', allow_early_resolve=True):
        il_ones = pl.full([B_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(il_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        for il_blk in pl.range(b_dim // B_TILE):
            il_b0 = il_blk * B_TILE
            cos_il[il_b0:il_b0 + B_TILE, 0:ROPE_HEAD_DIM] = pl.gather(cos_half[il_b0:il_b0 + B_TILE, 0:HALF_ROPE], dim=-1, index=il_dup_idx)
            sin_signed[il_b0:il_b0 + B_TILE, 0:ROPE_HEAD_DIM] = pl.mul(pl.gather(sin_half[il_b0:il_b0 + B_TILE, 0:HALF_ROPE], dim=-1, index=il_dup_idx), il_sign)
