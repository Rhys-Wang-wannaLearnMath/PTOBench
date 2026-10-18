import pypto.language as pl

from config import B, ROPE_HEAD_DIM, HALF_ROPE

@pl.jit.inline
def rope_interleave(
    cos_half: pl.Tensor[[B, HALF_ROPE], pl.FP32],
    sin_half: pl.Tensor[[B, HALF_ROPE], pl.FP32],
    cos_il: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
    sin_signed: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
):
    """Expand half-width cos/sin rows to the interleaved, sign-folded rope layout."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_interleave", allow_early_resolve=True):
        il_ones = pl.full([B, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(
            il_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        cos_il[0:B, 0:ROPE_HEAD_DIM] = pl.gather(
            cos_half[0:B, 0:HALF_ROPE], dim=-1, index=il_dup_idx)
        sin_signed[0:B, 0:ROPE_HEAD_DIM] = pl.mul(
            pl.gather(sin_half[0:B, 0:HALF_ROPE], dim=-1, index=il_dup_idx), il_sign)
