import torch

from config import CASE_CONFIG, TP_SIZE


VOCAB_PER_TP = CASE_CONFIG.vocab_size // TP_SIZE


def golden(tensors):
    hidden_states = tensors["hidden_states"].float()
    lm_head_weight = tensors["lm_head_weight"].float()
    shard_logits = []
    for rank in range(lm_head_weight.shape[0]):
        shard_logits.append(torch.matmul(hidden_states[rank], lm_head_weight[rank].t()))
    full_logits = [torch.cat(shard_logits, dim=-1) for _ in range(TP_SIZE)]
    tensors["logits"][:] = torch.stack(full_logits, dim=0)
