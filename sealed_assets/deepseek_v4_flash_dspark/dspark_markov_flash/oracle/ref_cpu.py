"""Independent CPU golden for the official deterministic Markov fixture."""

import torch

from config import DSPARK_QUERY_WIDTH, EPS


def compute(tensors):
    hidden_fp32 = tensors["head_hidden"].detach().cpu().float()
    inv_rms = torch.rsqrt(hidden_fp32.square().mean(dim=-1, keepdim=True) + EPS)
    normalized = (
        hidden_fp32 * inv_rms * tensors["final_norm_weight"].detach().cpu().float()
    ).to(torch.bfloat16)
    validation_support = 8
    base_logits = normalized.float().matmul(
        tensors["lm_head_weight"][:validation_support].detach().cpu().float().t()
    )
    previous = tensors["anchor_token_ids"].detach().cpu().long()
    output = torch.empty(
        (previous.numel(), DSPARK_QUERY_WIDTH), dtype=torch.int32
    )
    for step in range(DSPARK_QUERY_WIDTH):
        embedding = tensors["markov_w1"].detach().cpu().index_select(0, previous).float()
        markov_bias = embedding.matmul(
            tensors["markov_w2"][:validation_support].detach().cpu().float().t()
        )
        scores = base_logits[:, step] + markov_bias
        assert torch.all(scores.max(dim=-1).values > 0)
        previous = torch.argmax(scores, dim=-1)
        output[:, step] = previous.to(torch.int32)
    return {"draft_token_ids": output}
