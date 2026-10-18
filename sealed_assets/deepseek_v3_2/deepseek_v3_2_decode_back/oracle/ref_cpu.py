"""Case-local CPU Oracle for DeepSeek V3.2 decode back."""

import torch

EPS = 1.0e-6
HIDDEN_INV = 1.0 / 7168
K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 256


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"decode_back has no scalars: {sorted(scalars)}")
    hidden_states = inputs["hidden_states"]
    node_id = int(inputs["node_id_t"][0].item())
    combine_buf = inputs["combine_buf"]
    wo = inputs["wo"]
    post_rms_weight = inputs["post_rms_weight"]
    w_gate = inputs["w_gate"]
    w_up = inputs["w_up"]
    w_down = inputs["w_down"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inter_size = w_gate.shape[1]
    attn_out_size = combine_buf.shape[2]
    combined = combine_buf[node_id]

    resid1 = torch.zeros(batch, hidden_size, dtype=torch.float32)
    for o0 in range(0, hidden_size, Q_OUT_CHUNK):
        width = min(Q_OUT_CHUNK, hidden_size - o0)
        o_acc = torch.zeros(batch, width, dtype=torch.float32)
        for k0 in range(0, attn_out_size, K_CHUNK):
            o_acc += (
                combined[:, k0:k0 + K_CHUNK].float()
                @ wo[k0:k0 + K_CHUNK, o0:o0 + width].float()
            )
        resid1[:, o0:o0 + width] = o_acc + hidden_states[:, o0:o0 + width].float()

    sq_sum = torch.zeros(1, batch, dtype=torch.float32)
    for k0 in range(0, hidden_size, K_CHUNK):
        x_chunk = resid1[:, k0:k0 + K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=-1, keepdim=True).T
    inv_rms = torch.reciprocal(torch.sqrt(sq_sum * HIDDEN_INV + EPS))

    post_norm_bf16 = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    for k0 in range(0, hidden_size, K_CHUNK):
        x_chunk = resid1[:, k0:k0 + K_CHUNK]
        gamma = post_rms_weight[:, k0:k0 + K_CHUNK].float()
        post_norm_bf16[:, k0:k0 + K_CHUNK] = (x_chunk * inv_rms.T * gamma).bfloat16()

    mlp_bf16 = torch.zeros(batch, inter_size, dtype=torch.bfloat16)
    for o0 in range(0, inter_size, MLP_OUT_CHUNK):
        width = min(MLP_OUT_CHUNK, inter_size - o0)
        gate_acc = torch.zeros(batch, width, dtype=torch.float32)
        up_acc = torch.zeros(batch, width, dtype=torch.float32)
        for k0 in range(0, hidden_size, K_CHUNK):
            post_chunk = post_norm_bf16[:, k0:k0 + K_CHUNK].float()
            gate_acc += post_chunk @ w_gate[k0:k0 + K_CHUNK, o0:o0 + width].float()
            up_acc += post_chunk @ w_up[k0:k0 + K_CHUNK, o0:o0 + width].float()
        sigmoid = torch.reciprocal(torch.exp(-gate_acc) + 1.0)
        mlp_bf16[:, o0:o0 + width] = (gate_acc * sigmoid * up_acc).bfloat16()

    out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    for d0 in range(0, hidden_size, K_CHUNK):
        width = min(K_CHUNK, hidden_size - d0)
        down_acc = torch.zeros(batch, width, dtype=torch.float32)
        for o0 in range(0, inter_size, MLP_OUT_CHUNK):
            down_acc += (
                mlp_bf16[:, o0:o0 + MLP_OUT_CHUNK].float()
                @ w_down[o0:o0 + MLP_OUT_CHUNK, d0:d0 + width].float()
            )
        out[:, d0:d0 + width] = (down_acc + resid1[:, d0:d0 + width]).bfloat16()
    return {"out": out.contiguous()}

