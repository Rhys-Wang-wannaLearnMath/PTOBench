"""Markov embedding and full-vocabulary projection."""

import pypto.language as pl

from config import DSPARK_MARKOV_RANK as MARKOV_RANK

T_DYN = pl.dynamic("MARKOV_HEAD_T_DYN")
VOCAB_DYN = pl.dynamic("MARKOV_HEAD_VOCAB_DYN")

T_TILE = 16
VOCAB_TILE = 128
SPMD_BLOCKS = 48


@pl.jit.inline
def markov_head(
    token_ids: pl.Tensor[[T_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    logits_bias: pl.Tensor[[T_DYN, VOCAB_DYN], pl.FP32],
    markov_embed: pl.Tensor[[T_DYN, MARKOV_RANK], pl.BF16],
):
    t_dim = pl.tensor.dim(token_ids, 0)
    t_linear = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE

    for token_idx in pl.spmd(t_dim, name_hint="markov_embedding", allow_early_resolve=True):
        token_id = pl.read(token_ids, [token_idx])
        token_row = pl.cast(token_id, target_type=pl.INDEX)
        markov_embed[token_idx : token_idx + 1, 0:MARKOV_RANK] = markov_w1[
            token_row : token_row + 1, 0:MARKOV_RANK
        ]

    vocab_dim = pl.tensor.dim(markov_w2, 0)
    work_items = (t_linear // T_TILE) * (vocab_dim // VOCAB_TILE)
    for block in pl.spmd(SPMD_BLOCKS, name_hint="markov_logits"):
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            t0 = (work_idx // (vocab_dim // VOCAB_TILE)) * T_TILE
            vocab0 = (work_idx % (vocab_dim // VOCAB_TILE)) * VOCAB_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            embed_bf16 = pl.slice(
                markov_embed, [T_TILE, MARKOV_RANK], [t0, 0], valid_shape=[valid_rows, MARKOV_RANK]
            )
            weight_bf16 = markov_w2[vocab0 : vocab0 + VOCAB_TILE, 0:MARKOV_RANK]
            logits_tile = pl.matmul(embed_bf16, weight_bf16, b_trans=True, out_dtype=pl.FP32)
            logits_valid = pl.set_validshape(logits_tile, valid_rows, VOCAB_TILE)
            logits_bias[t0 : t0 + T_TILE, vocab0 : vocab0 + VOCAB_TILE] = logits_valid

    return logits_bias, markov_embed
