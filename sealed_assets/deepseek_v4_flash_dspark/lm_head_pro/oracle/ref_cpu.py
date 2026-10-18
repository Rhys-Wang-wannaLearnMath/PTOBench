import torch

from config import D, DP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD, TP_SIZE, VOCAB


def compute(values):
    hidden = values["hidden_states"].detach().cpu().float()
    weight = values["lm_head_weight"].detach().cpu().float()
    full_weight = torch.cat([weight[rank] for rank in range(TP_SIZE)], dim=0)
    logits = torch.zeros((DP_SIZE, MAX_LOGIT_ROWS, VOCAB), dtype=torch.float32)
    indices = values["logit_row_indices"].detach().cpu()
    for owner in range(DP_SIZE):
        selected = torch.zeros((MAX_LOGIT_ROWS, D), dtype=torch.float32)
        for row in range(MAX_LOGIT_ROWS):
            source = int(indices[owner, row])
            if source >= 0:
                source = min(source, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner, source])
        logits[owner] = selected.matmul(full_weight.t())
    sampled_ids = torch.zeros(
        (DP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), dtype=torch.int32
    )
    sampled_ids[:, :, 0] = logits.argmax(dim=-1).to(torch.int32)
    return {"logits": logits, "sampled_ids": sampled_ids}
