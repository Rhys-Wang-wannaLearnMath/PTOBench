"""Case-local CPU Oracle for Batch 07 DeepSeek V4 decode attention."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_ORACLE_DIR = Path(__file__).resolve().parent
_CASE_DIR = Path(__file__).resolve().parents[1]
_BENCH_ROOT = _CASE_DIR.parents[2]

if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

def _helper(name):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"oracle_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous_config = sys.modules.get("config")
    sys.modules["config"] = _config_module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_config
    return module

import config as _config_module

from config import (
    CASE_CONFIG as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
COMPRESS_RATIO = 128  # HCA
OVERLAP = COMPRESS_RATIO == 4   # always False for HCA
COFF = 1 + int(OVERLAP)         # always 1 for HCA
MAIN_OUT_DIM = COFF * HEAD_DIM
ORI_MAX_BLOCKS = 1                  # WIN==BLOCK_SIZE → 1 ori block per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS  # ori KV pool size (matches sparse_attn ORI_BLOCK_NUM)
CMP_MAX_BLOCKS = 64                 # matches sparse_attn CMP_MAX_BLOCKS
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS  # cmp KV pool size (shared with the compressor's cmp_kv_cache)
COMPRESS_STATE_MAX_BLOCKS = 64
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
SPARSE_IDX_TOPK = M.index_topk             # sparse_attn module's IDX_TOPK (static shape contract)
CMP_TOPK_CAPACITY = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_TOPK = min(CMP_TOPK_CAPACITY, SPARSE_IDX_TOPK)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK        # sparse_attn module's TOPK (= 640 for demo)
START_POS = 128
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE

def golden_attention_hca(tensors):
    """End-to-end orchestration for the ratio=128 (HCA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==128 path: main compressor only,
    no indexer, compress_topk_idxs computed deterministically) + Block.hc_post."""
    import torch

    golden_hc_pre = _helper("hc_pre").golden_hc_pre
    qkv = _helper("decode_qkv_proj_rope")
    golden_attn_norm, golden_qkv_proj_rope = qkv.golden_attn_norm, qkv.golden_qkv_proj_rope
    golden_compressor = _helper("decode_compressor_ratio128").golden_compressor
    golden_sparse_attn = _helper("decode_sparse_attn").golden_sparse_attn
    golden_hc_post = _helper("hc_post").golden_hc_post

    # ---- Block.hc_pre ----
    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # ===== Attention.forward, ratio==128 branch =====
    start_pos_t = tensors["start_pos"].to(torch.int64)
    bsz, seqlen, _ = x_mixed.shape
    win = WIN
    ratio = COMPRESS_RATIO
    rd = ROPE_HEAD_DIM

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_T = torch.empty(T, rd, dtype=freqs_cos.dtype)
    rope_sin_T = torch.empty(T, rd, dtype=freqs_sin.dtype)
    half_rd = rd // 2
    cmp_cos = torch.empty(B, half_rd, dtype=torch.float32)
    cmp_sin = torch.empty(B, half_rd, dtype=torch.float32)
    cmp_start_pos = start_pos_t.to(torch.int32).contiguous()
    for b in range(B):
        start_pos_b = int(start_pos_t[b].item())
        for s in range(S):
            t = b * S + s
            pos = start_pos_b + s
            rope_cos_T[t] = freqs_cos[pos]
            rope_sin_T[t] = freqs_sin[pos]
        cmp_pos_b = start_pos_b + (ratio - (start_pos_b % ratio)) - ratio
        cmp_cos[b] = freqs_cos[cmp_pos_b, :half_rd].float()
        cmp_sin[b] = freqs_sin[cmp_pos_b, :half_rd].float()

    # q + win kv (W8A8 q_proj)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_attn_norm(x_mixed, tensors["attn_norm_w"])
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_T,
        "rope_sin": rope_sin_T,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,                                                              # qr unused on HCA path
        "qr_scale": qr_scale,
    })

    # window topk + compress topk; HCA uses get_compress_topk_idxs.
    # Pad to sparse_attn's static SPARSE_TOPK; seqused_kv bounds the valid range.
    topk_idxs = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
    topk_idxs[:, :win] = torch.arange(win, dtype=torch.int32)
    offset = win
    topk_idxs[:, win:win + CMP_TOPK] = torch.arange(CMP_TOPK, dtype=torch.int32) + offset
    topk_idxs = topk_idxs.int()

    kv_cache = tensors["kv_cache"]
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]
    for t in range(T):
        b = t // S
        s = t % S
        start_pos_b = int(start_pos_t[b].item())
        ori_slot = (start_pos_b + s) % win
        blk_id = int(ori_block_table[b, ori_slot // BLOCK_SIZE].item())
        intra = ori_slot % BLOCK_SIZE
        kv_cache[blk_id, intra, 0] = kv[t]

    cmp_kv_proj = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    golden_compressor({
        "x": x_normed,
        "kv": cmp_kv_proj,
        "compress_state": tensors["compress_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": cmp_cos,
        "sin": cmp_sin,
        "cmp_kv_cache": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "start_pos": cmp_start_pos,
    })

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": ori_block_table,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "cmp_sparse_indices": topk_idxs,
        "attn_sink": tensors["attn_sink"],
        "seqused_kv": tensors["seqused_kv"],
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # ===== Block.hc_post =====
    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y

def compute(inputs, scalars):
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors.update(dict(scalars))
    tensors["x_out"] = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_attention_hca(tensors)
    return {"x_out": tensors["x_out"].detach().cpu().contiguous()}
