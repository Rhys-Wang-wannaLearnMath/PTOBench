"""Case-local CPU Oracle for DeepSeek V3.2 prefill back."""

import torch

EPS = 1.0e-6
K_CHUNK = 128
MLP_OUT_CHUNK = 128


def _chunked_bf16_matmul(a_bf16, b_bf16, k_chunk=K_CHUNK):
    k_size = a_bf16.shape[-1]
    acc = torch.zeros(*a_bf16.shape[:-1], b_bf16.shape[-1], dtype=torch.float32)
    for k0 in range(0, k_size, k_chunk):
        k1 = min(k0 + k_chunk, k_size)
        acc += torch.matmul(a_bf16[..., k0:k1].float(), b_bf16[k0:k1, :].float())
    return acc


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"prefill_back has no scalars: {sorted(scalars)}")
    hidden_states = inputs["hidden_states"]
    seq_lens = inputs["seq_lens"]
    combine_buf = inputs["combine_buf"]
    wo = inputs["wo"]
    post_rms_weight = inputs["post_rms_weight"]
    w_gate = inputs["w_gate"]
    w_up = inputs["w_up"]
    w_down = inputs["w_down"]

    out = torch.zeros_like(hidden_states)
    post_rms_f = post_rms_weight.float()
    for b in range(hidden_states.shape[0]):
        seq_len_b = int(seq_lens[b].item())
        active = slice(0, seq_len_b)
        attn_bf16 = combine_buf[0, b, active, :]
        hs = hidden_states[b, active, :].float()
        resid1 = _chunked_bf16_matmul(attn_bf16, wo) + hs
        inv_rms = torch.rsqrt(resid1.pow(2).mean(dim=-1, keepdim=True) + EPS)
        normed_bf16 = (resid1 * inv_rms * post_rms_f).bfloat16()
        gate = _chunked_bf16_matmul(normed_bf16, w_gate)
        up = _chunked_bf16_matmul(normed_bf16, w_up)
        mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
        down = _chunked_bf16_matmul(mlp_bf16, w_down, k_chunk=MLP_OUT_CHUNK)
        out[b, active, :] = (down + resid1).bfloat16()
    return {"out": out.contiguous()}

