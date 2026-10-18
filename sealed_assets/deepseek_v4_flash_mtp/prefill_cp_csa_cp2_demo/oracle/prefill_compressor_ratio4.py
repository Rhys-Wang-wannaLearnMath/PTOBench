from config import BLOCK_SIZE, CSA_STATE_PHYSICAL_BLOCKS, CASE_CONFIG as M

EPS = M.rms_norm_eps

D = M.hidden_size

HEAD_DIM = M.head_dim

ROPE_HEAD_DIM = M.qk_rope_head_dim

NOPE_HEAD_DIM = M.nope_head_dim

MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 4

OVERLAP = COMPRESS_RATIO == 4

COFF = 1 + int(OVERLAP)

OUT_DIM = COFF * HEAD_DIM

STATE_LEN = COFF * COMPRESS_RATIO

B = 1

S = 128

T = B * S

CSA_STATE_BLOCK_SIZE = 4

CSA_STATE_BLOCK_NUM = CSA_STATE_PHYSICAL_BLOCKS

COMPRESS_STATE_DIM = 2 * OUT_DIM

def golden_prefill_compressor_ratio4(tensors):
    """Packed token-major torch reference for ratio-4 prefill compressor."""
    import torch
    x = tensors['x'].view(T, D).float()
    compress_state_flat = tensors['compress_state'].view(CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors['compress_state_block_table']
    wkv = tensors['wkv'].float()
    wgate = tensors['wgate'].float()
    ape = tensors['ape']
    norm_w = tensors['norm_w']
    cmp_kv = tensors['cmp_kv']
    cache_rows = cmp_kv.view(cmp_kv.shape[0] * BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    position_ids = tensors['position_ids']
    kv_proj = x @ wkv.t()
    score_proj = x @ wgate.t()

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * CSA_STATE_BLOCK_SIZE + intra
    for token_id in range(int(tensors['num_tokens'])):
        dst_row = int(tensors['cmp_slot_mapping'][token_id].item())
        if dst_row < 0:
            continue
        write_pos = int(position_ids[token_id].item())
        cur_start = write_pos + 1 - COMPRESS_RATIO
        prev_start = cur_start - COMPRESS_RATIO
        pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
        pool_score = torch.full((STATE_LEN, HEAD_DIM), float('-inf'), dtype=torch.float32)
        for s in range(COMPRESS_RATIO):
            prev_abs = prev_start + s
            if write_pos >= 2 * COMPRESS_RATIO - 1:
                prev_row = state_row(prev_abs)
                if prev_row >= 0:
                    pool_kv[s] = kv_state_flat[prev_row, :HEAD_DIM]
                    pool_score[s] = score_state_flat[prev_row, :HEAD_DIM]
            cur_abs = cur_start + s
            cur_row = state_row(cur_abs)
            if cur_row >= 0:
                pool_kv[COMPRESS_RATIO + s] = kv_state_flat[cur_row, HEAD_DIM:OUT_DIM]
                pool_score[COMPRESS_RATIO + s] = score_state_flat[cur_row, HEAD_DIM:OUT_DIM]
        for t in range(int(tensors['num_tokens'])):
            pos = int(position_ids[t].item())
            if pos < prev_start or pos > write_pos:
                continue
            if pos < cur_start:
                pool_slot = pos - prev_start
                col0 = 0
            else:
                pool_slot = COMPRESS_RATIO + pos - cur_start
                col0 = HEAD_DIM
            ape_slot = pos % COMPRESS_RATIO
            pool_kv[pool_slot] = kv_proj[t, col0:col0 + HEAD_DIM]
            pool_score[pool_slot] = score_proj[t, col0:col0 + HEAD_DIM] + ape[ape_slot, col0:col0 + HEAD_DIM]
        init_slot = STATE_LEN - 1
        mi = pool_score[init_slot:init_slot + 1].clone()
        li = torch.exp(mi - mi)
        oi = pool_kv[init_slot:init_slot + 1].clone()
        for slot_i in range(STATE_LEN - 1):
            if slot_i < COMPRESS_RATIO and write_pos < 2 * COMPRESS_RATIO - 1:
                continue
            slot_score = pool_score[slot_i:slot_i + 1]
            slot_kv = pool_kv[slot_i:slot_i + 1]
            mi_next = torch.maximum(mi, slot_score)
            alpha = torch.exp(mi - mi_next)
            beta = torch.exp(slot_score - mi_next)
            li = alpha * li + beta
            oi = oi * alpha + slot_kv * beta
            mi = mi_next
        pooled = oi / li
        inv_rms = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = pooled * inv_rms * norm_w.float().view(1, HEAD_DIM)
        rope_pair = normed[..., NOPE_HEAD_DIM:HEAD_DIM].unflatten(-1, (-1, 2))
        rope_even = rope_pair[..., 0]
        rope_odd = rope_pair[..., 1]
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors['freqs_cos'][cmp_pos:cmp_pos + 1, 0:ROPE_HEAD_DIM // 2].float()
        sin = tensors['freqs_sin'][cmp_pos:cmp_pos + 1, 0:ROPE_HEAD_DIM // 2].float()
        rot_even = rope_even * cos - rope_odd * sin
        rot_odd = rope_even * sin + rope_odd * cos
        normed[:, NOPE_HEAD_DIM:HEAD_DIM] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cache_rows[dst_row] = normed.to(torch.bfloat16)[0]
    for t in range(int(tensors['num_tokens'])):
        pos = int(tensors['position_ids'][t].item())
        dst_row = int(tensors['state_slot_mapping'][t].item())
        if dst_row < 0:
            continue
        ape_slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors['ape'][ape_slot]
    tensors['cmp_kv'][:] = cmp_kv
