"""Independent CPU Oracle for DeepSeek-V4 sliding-window decode."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from config import (
    BLOCK_SIZE,
    CASE_CONFIG as M,
    DECODE_BATCH,
    DECODE_SEQ,
)


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
ORI_MAX_BLOCKS = 1
TOPK = WIN
SPARSE_IDX_TOPK = M.index_topk
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK
SPARSE_CMP_MAX_BLOCKS = 64


def _helper(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"decode_swa_oracle_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Oracle helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def golden_attention_swa(tensors: dict[str, torch.Tensor]) -> None:
    """Execute the frozen one-step SWA semantics independently with Torch."""
    golden_hc_pre = _helper("hc_pre").golden_hc_pre
    qkv = _helper("decode_qkv_proj_rope")
    golden_sparse_attn = _helper("decode_sparse_attn").golden_sparse_attn
    golden_hc_post = _helper("hc_post").golden_hc_post

    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT)
    golden_hc_pre(
        {
            "x": tensors["x_hc"],
            "hc_fn": tensors["hc_attn_fn"],
            "hc_scale": tensors["hc_attn_scale"],
            "hc_base": tensors["hc_attn_base"],
            "x_mixed": x_mixed,
            "post": post_t,
            "comb": comb_t,
        }
    )

    start_pos = tensors["start_pos"].to(torch.int64)
    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos = torch.empty(T, ROPE_HEAD_DIM, dtype=freqs_cos.dtype)
    rope_sin = torch.empty(T, ROPE_HEAD_DIM, dtype=freqs_sin.dtype)
    for batch in range(B):
        position = int(start_pos[batch].item())
        for sequence in range(S):
            token = batch * S + sequence
            rope_cos[token] = freqs_cos[position + sequence]
            rope_sin[token] = freqs_sin[position + sequence]

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = qkv.golden_attn_norm(x_mixed, tensors["attn_norm_w"])
    qkv.golden_qkv_proj_rope(
        {
            "x": x_normed,
            "wq_a": tensors["wq_a"],
            "wq_b": tensors["wq_b"],
            "wq_b_scale": tensors["wq_b_scale"],
            "wkv": tensors["wkv"],
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "gamma_cq": tensors["gamma_cq"],
            "gamma_ckv": tensors["gamma_ckv"],
            "q": q,
            "kv": kv,
            "qr": qr,
            "qr_scale": qr_scale,
        }
    )

    topk = torch.full((T, TOPK), -1, dtype=torch.int32)
    topk[:, :WIN] = torch.arange(WIN, dtype=torch.int32)

    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]
    for token in range(T):
        batch = token // S
        sequence = token % S
        position = int(start_pos[batch].item())
        slot = (position + sequence) % WIN
        block = int(block_table[batch, slot // BLOCK_SIZE].item())
        kv_cache[block, slot % BLOCK_SIZE, 0] = kv[token]

    sparse_topk = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
    sparse_topk[:, :WIN] = topk
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    cmp_kv = torch.zeros(
        B * SPARSE_CMP_MAX_BLOCKS,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    cmp_block_table = torch.full(
        (B, SPARSE_CMP_MAX_BLOCKS), -1, dtype=torch.int32
    )
    golden_sparse_attn(
        {
            "q": q,
            "ori_kv": kv_cache,
            "ori_block_table": block_table[:, :ORI_MAX_BLOCKS],
            "cmp_kv": cmp_kv,
            "cmp_block_table": cmp_block_table,
            "cmp_sparse_indices": sparse_topk,
            "attn_sink": tensors["attn_sink"],
            "seqused_kv": tensors["seqused_kv"],
            "freqs_cos": rope_cos,
            "freqs_sin": rope_sin,
            "wo_a": tensors["wo_a"],
            "wo_b": tensors["wo_b"],
            "wo_b_scale": tensors["wo_b_scale"],
            "attn_out": attn_out,
        }
    )

    output = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post(
        {
            "x": attn_out.view(B, S, D),
            "residual": tensors["x_hc"],
            "post": post_t,
            "comb": comb_t,
            "y": output,
        }
    )
    tensors["x_out"][:] = output


def compute(inputs: dict[str, torch.Tensor], scalars: dict) -> dict[str, torch.Tensor]:
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors.update(scalars)
    tensors["x_out"] = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_attention_swa(tensors)
    return {"x_out": tensors["x_out"].detach().cpu().contiguous()}
