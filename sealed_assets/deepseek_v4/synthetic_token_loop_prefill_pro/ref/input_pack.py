import pypto.language as pl
from config import CASE_CONFIG as M
TOKEN_DYN = pl.dynamic('PACK_X_HC_TOKEN_DYN')
VOCAB_DYN = pl.dynamic('PACK_X_HC_VOCAB_DYN')
from config import D
from config import HC_MULT
HIDDEN_TILE = 512
SPMD_BLOCKS = 48

@pl.jit.inline
def pack_x_hc(input_ids: pl.Tensor[[TOKEN_DYN], pl.INT64], embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16], x_hc: pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32]) -> pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32]:
    """Look up token embeddings and replicate them across the HC lanes."""
    token_count = pl.tensor.dim(input_ids, 0)
    vocab_rows = pl.tensor.dim(embed_weight, 0)
    x_hc_flat = pl.reshape(x_hc, [token_count * HC_MULT, D])
    work_items = token_count * (D // HIDDEN_TILE)
    for block in pl.spmd(SPMD_BLOCKS, name_hint='pack_x_hc'):
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            token_idx = work_idx // (D // HIDDEN_TILE)
            hidden_offset = work_idx % (D // HIDDEN_TILE) * HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            safe_token_id = pl.max(pl.min(token_id, vocab_rows - 1), 0)
            token_row = pl.cast(safe_token_id, target_type=pl.INDEX)
            hidden_chunk = pl.cast(embed_weight[token_row:token_row + 1, hidden_offset:hidden_offset + HIDDEN_TILE], target_type=pl.FP32)
            for hc_idx in pl.range(HC_MULT):
                x_hc_row = token_idx * HC_MULT + hc_idx
                x_hc_flat[x_hc_row:x_hc_row + 1, hidden_offset:hidden_offset + HIDDEN_TILE] = hidden_chunk
    return x_hc
