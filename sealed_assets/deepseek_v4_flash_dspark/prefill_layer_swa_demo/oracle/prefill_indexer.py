# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ORACLE closure for deepseek_v4_flash_dspark.prefill_layer_swa_demo; fixed bindings from public config."""
from config import BLOCK_SIZE
from config import FP32_NEG_INF
from config import IDX_CACHE_BLOCK_NUM
from config import IDX_HEAD_DIM
from config import IDX_N_HEADS
from config import IDX_TOPK
from config import ROPE_HEAD_DIM
from config import WEIGHTS_SCALE
from prefill_indexer_compressor import golden_prefill_indexer_compressor
from utils import int8_quant_per_row

PREFILL_DENSE_TILE = 512
COMPRESS_RATIO = 4
RAW_TOKEN_CAP = 8192
INDEXER_SCORE_CAP = RAW_TOKEN_CAP // COMPRESS_RATIO
INDEXER_TOPK_CAP = min(IDX_TOPK, INDEXER_SCORE_CAP)
Q_TILE = 128

def topk_prefix_contract_error(topk_indices, position_ids, num_tokens=None):
    """Return an error string if the top-k prefix contract is broken."""
    import torch

    physical_tokens = int(topk_indices.shape[0])
    if num_tokens is None:
        num_tokens = physical_tokens
    elif hasattr(num_tokens, "item"):
        num_tokens = num_tokens.item()
    num_tokens = int(num_tokens)
    for t in range(physical_tokens):
        row = topk_indices[t]
        if t >= num_tokens:
            non_padding = int((row != -1).count_nonzero().item())
            if non_padding:
                return f"inactive top-k row {t} contains {non_padding} non--1 entries"
            continue
        visible_candidates = min(
            int((int(position_ids[t].item()) + 1) // COMPRESS_RATIO),
            INDEXER_SCORE_CAP,
        )
        selected = min(INDEXER_TOPK_CAP, visible_candidates)
        prefix = row[:selected]
        if selected:
            out_of_range = int(((prefix < 0) | (prefix >= visible_candidates)).count_nonzero().item())
            if out_of_range:
                return (
                    f"top-k row {t} has {out_of_range} entries outside "
                    f"[0, {visible_candidates}) in its selected prefix"
                )
            unique_count = int(torch.unique(prefix).numel())
            if unique_count != selected:
                return f"top-k row {t} selected prefix has {unique_count}/{selected} unique entries"
        tail_non_padding = int((row[selected:] != -1).count_nonzero().item())
        if tail_non_padding:
            return f"top-k row {t} tail contains {tail_non_padding} non--1 entries"
    return None


def golden_prefill_indexer_core(tensors):
    # Oracle-local dependency bound during isolated module load.
    import torch

    token_count = int(tensors["x"].shape[0])
    compressor_tensors = {
        "x": tensors["x"],
        "kv": torch.zeros(
            max(1, (token_count + COMPRESS_RATIO - 1) // COMPRESS_RATIO),
            IDX_HEAD_DIM,
            dtype=torch.int8,
        ),
        "compress_state": tensors["inner_compress_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "hadamard": tensors["hadamard"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "idx_block_table": tensors["idx_block_table"],
        "position_ids": tensors["position_ids"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    }
    golden_prefill_indexer_compressor(compressor_tensors)
    tensors["idx_kv_cache"][:] = compressor_tensors["idx_kv_cache"]
    tensors["idx_kv_scale"][:] = compressor_tensors["idx_kv_scale"]

    # --- Real lightning-indexer score + per-token causal-masked top-k ---
    # Ports the official model.py Indexer.forward(start_pos==0) branch (and the deleted #505^
    # prefill indexer): score each token's query against the compressed index KV through the
    # W8A8C16 int8 path, causal-mask each token to the positions it can reach ((pos+1)//ratio),
    # then top-k. Replaces the old placeholder (sequential arange+offset, i.e. the dense
    # get_compress_topk_idxs pattern, which never exercised real selection).
    position_ids = tensors["position_ids"].long()
    rd = ROPE_HEAD_DIM
    cmp_topk_indices = torch.full((token_count, IDX_TOPK), -1, dtype=torch.int32)
    score_full = torch.full(
        (token_count, INDEXER_SCORE_CAP),
        FP32_NEG_INF,
        dtype=torch.float32,
    )
    visible = ((position_ids + 1) // COMPRESS_RATIO).clamp(max=INDEXER_SCORE_CAP)
    max_visible = int(visible.max().item())
    if max_visible == 0:
        return cmp_topk_indices, score_full

    # Q: int8 qr x int8 wq_b -> dequant -> per-token interleaved RoPE -> Hadamard rotation.
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    hadamard = tensors["hadamard"].float()

    # C8: the compressor already stored INT8 KV + a per-position dequant scale. Gather both in
    # compressed-position order through the paged block table (no score-time re-quant).
    cache_flat_i8 = tensors["idx_kv_cache"].reshape(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
    scale_flat = tensors["idx_kv_scale"].float().reshape(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, 1)
    idx_block_table = tensors["idx_block_table"]
    rows = [
        int(idx_block_table[c // BLOCK_SIZE].item()) * BLOCK_SIZE + (c % BLOCK_SIZE)
        for c in range(max_visible)
    ]
    kv_i8 = torch.stack([cache_flat_i8[r] for r in rows], dim=0).to(torch.int32)  # [max_visible, dim]
    kv_sc = torch.stack([scale_flat[r] for r in rows], dim=0).view(1, 1, max_visible)

    # Keep the golden bounded too: reproduce the kernel's 2048-token dense
    # tile, rather than materializing a full [8192, 8192] projection.
    col = torch.arange(max_visible).unsqueeze(0)
    for tile_base in range(0, token_count, PREFILL_DENSE_TILE):
        tile_end = min(tile_base + PREFILL_DENSE_TILE, token_count)
        tile_rows = tile_end - tile_base
        qr_tile = tensors["qr"][tile_base:tile_end]
        qr_scale_tile = tensors["qr_scale"][tile_base:tile_end].float()
        q_i32 = qr_tile.to(torch.int32) @ wq_b.to(torch.int32)
        q = (q_i32.float() * qr_scale_tile * wq_b_scale.view(1, -1)).view(
            tile_rows, IDX_N_HEADS, IDX_HEAD_DIM
        )
        q_pair = q[..., -rd:].unflatten(-1, (-1, 2))
        q0, q1 = q_pair[..., 0], q_pair[..., 1]
        cos_tile = tensors["cos"][tile_base:tile_end].float().view(tile_rows, 1, -1)
        sin_tile = tensors["sin"][tile_base:tile_end].float().view(tile_rows, 1, -1)
        y0 = (q0 * cos_tile - q1 * sin_tile).to(torch.bfloat16)
        y1 = (q0 * sin_tile + q1 * cos_tile).to(torch.bfloat16)
        q = torch.cat(
            [q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)],
            dim=-1,
        )
        q = q.to(torch.bfloat16).float() @ hadamard
        weights = (tensors["x"][tile_base:tile_end].float() @ tensors["weights_proj"].float()) * WEIGHTS_SCALE

        # W8A8C16 score, matching decode_indexer: per-row quantize q, INT32
        # matmul against pre-quantized KV, then dequantize both operands.
        q_i8, q_sc = int8_quant_per_row(q.reshape(tile_rows * IDX_N_HEADS, IDX_HEAD_DIM))
        q_i8 = q_i8.view(tile_rows, IDX_N_HEADS, IDX_HEAD_DIM).to(torch.int32)
        q_sc = q_sc.view(tile_rows, IDX_N_HEADS, 1)
        for wave_base in range(0, tile_rows, Q_TILE):
            wave_end = min(wave_base + Q_TILE, tile_rows)
            score_wave = torch.einsum(
                "thd,cd->thc",
                q_i8[wave_base:wave_end],
                kv_i8,
            ).float()
            score_wave = score_wave * q_sc[wave_base:wave_end] * kv_sc
            score_wave = (torch.relu(score_wave) * weights[wave_base:wave_end].unsqueeze(-1)).sum(dim=1)
            global_wave_base = tile_base + wave_base
            global_wave_end = tile_base + wave_end
            score_wave = score_wave.masked_fill(
                col >= visible[global_wave_base:global_wave_end].unsqueeze(1),
                FP32_NEG_INF,
            )
            score_full[global_wave_base:global_wave_end, :max_visible] = score_wave
            for wave_t in range(wave_end - wave_base):
                global_t = global_wave_base + wave_t
                k = int(min(INDEXER_TOPK_CAP, int(visible[global_t].item())))
                if k > 0:
                    sel = score_wave[wave_t].topk(k, dim=-1)[1]
                    cmp_topk_indices[global_t, :k] = sel.to(torch.int32)
    return cmp_topk_indices, score_full


