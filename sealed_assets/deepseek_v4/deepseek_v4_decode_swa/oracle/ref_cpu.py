"""Case-local CPU Oracle for Batch 07 DeepSeek V4 decode attention."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from deepseek_v4_decode_hc_post import golden_deepseek_v4_decode_hc_post
from deepseek_v4_decode_hc_pre import golden_deepseek_v4_decode_hc_pre
from deepseek_v4_decode_qkv_proj_rope import golden_deepseek_v4_decode_qkv_proj_rope

_WITH_PROJ_PATH = Path(__file__).resolve().parent / "deepseek_v4_decode_sparse_attn.py"
_WITH_PROJ_SPEC = importlib.util.spec_from_file_location(
    "_deepseek_v4_decode_swa_oracle_sparse_attn",
    _WITH_PROJ_PATH,
)
if _WITH_PROJ_SPEC is None or _WITH_PROJ_SPEC.loader is None:
    raise ImportError(f'cannot load frozen helper: {_WITH_PROJ_PATH}')
_WITH_PROJ_MOD = importlib.util.module_from_spec(_WITH_PROJ_SPEC)
_WITH_PROJ_SPEC.loader.exec_module(_WITH_PROJ_MOD)
golden_deepseek_v4_decode_sparse_attn_with_proj = (
    _WITH_PROJ_MOD.golden_deepseek_v4_decode_sparse_attn_with_proj
)

from config import (
    B,
    S,
    T,
    D,
    H,
    HEAD_DIM,
    ROPE_HEAD_DIM,
    Q_LORA,
    WIN,
    HC_MULT,
    O_GROUPS,
    BLOCK_SIZE,
    ORI_MAX_BLOCKS,
    TOPK,
    SPARSE_TOPK,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_CMP_BLOCK_NUM,
)

def golden_deepseek_v4_decode_swa(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    import torch

    # ---- Block.hc_pre (model.py:691) ----
    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT)
    golden_deepseek_v4_decode_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # ===== Attention.forward (model.py:484-543), ratio==0 branch =====
    start_pos = int(tensors["start_pos"])
    bsz, seqlen, _ = x_mixed.shape
    win = WIN
    rd = ROPE_HEAD_DIM

    if start_pos == 0:
        return  # prefill — decode-only orchestration skips

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    step_cos = freqs_cos[start_pos:start_pos + 1]                            # [1, rd]
    step_sin = freqs_sin[start_pos:start_pos + 1]
    rope_cos_T = step_cos.expand(T, rd).contiguous()
    rope_sin_T = step_sin.expand(T, rd).contiguous()

    # q + win kv (model.py:495-504)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    golden_deepseek_v4_decode_qkv_proj_rope({
        "x": x_mixed,
        "norm_w": tensors["attn_norm_w"],
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
        "qr": qr,                                                              # qr unused on SWA path
        "qr_scale": qr_scale,
    })

    # window topk only (model.py:507; ratio==0 skips lines 508-514)
    topk_idxs = torch.full((T, TOPK), -1, dtype=torch.int32)
    topk_idxs[:, :win] = torch.arange(win, dtype=torch.int32)

    # ori_kv scatter (model.py:530)
    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]
    ori_slot = start_pos % win
    for b in range(B):
        blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
        intra = ori_slot % BLOCK_SIZE
        kv_cache[blk_id, intra, 0] = kv[b]

    # sparse_attn (model.py:533); window-only uses the full sparse_attn topk contract with an empty cmp tail.
    sparse_topk = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
    sparse_topk[:, :WIN] = topk_idxs
    seqused_kv = tensors["seqused_kv"]
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    cmp_kv_dummy = torch.zeros(SPARSE_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    cmp_block_table_dummy = torch.full((B, SPARSE_CMP_MAX_BLOCKS), -1, dtype=torch.int32)
    golden_deepseek_v4_decode_sparse_attn_with_proj({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": block_table[:, :ORI_MAX_BLOCKS],
        "cmp_kv": cmp_kv_dummy,
        "cmp_block_table": cmp_block_table_dummy,
        "cmp_sparse_indices": sparse_topk,
        "attn_sink": tensors["attn_sink"],
        "seqused_kv": seqused_kv.view(B),
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # ===== Block.hc_post (model.py:694) =====
    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_deepseek_v4_decode_hc_post({
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
    golden_deepseek_v4_decode_swa(tensors)
    return {"x_out": tensors["x_out"].detach().cpu().contiguous()}
