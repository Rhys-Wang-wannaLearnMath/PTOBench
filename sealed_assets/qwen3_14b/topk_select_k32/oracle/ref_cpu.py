import torch

from config import FP32_NEG_INF, REAL_VOCAB, TOPK


def compute(inputs, scalars):
    logits = inputs["logits"]
    sampling_control = inputs["sampling_control"]
    active_batch = int(sampling_control[0].item())
    selection_k = int(sampling_control[1].item())
    real_vocab = min(REAL_VOCAB, logits.shape[1])

    topk_values = torch.full(
        (logits.shape[0], TOPK),
        FP32_NEG_INF,
        dtype=torch.float32,
    )
    topk_indices = torch.zeros((logits.shape[0], TOPK), dtype=torch.int32)
    active_logits = logits[:active_batch, :real_vocab].float()

    if selection_k == 1:
        token_ids = torch.argmax(active_logits, dim=-1)
        values = torch.gather(active_logits, dim=-1, index=token_ids[:, None])
        topk_values[:active_batch, :1] = values
        topk_indices[:active_batch, :1] = token_ids[:, None].to(torch.int32)
    else:
        values, token_ids = torch.topk(active_logits, TOPK, dim=-1, largest=True, sorted=True)
        topk_values[:active_batch] = values
        topk_indices[:active_batch] = token_ids.to(torch.int32)

    return {
        "topk_values": topk_values,
        "topk_indices": topk_indices,
    }
