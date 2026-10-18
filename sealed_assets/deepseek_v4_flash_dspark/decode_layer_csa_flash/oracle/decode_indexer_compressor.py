"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_csa_flash."""

from config import CASE_CONFIG as M
from config import COMPRESS_RATIO
from config import INT8_AMAX_EPS
from config import DECODE_SEQ
from config import INT8_SCALE_MAX
from config import C4A_COMPRESSOR_BLOCK_SIZE
from config import BLOCK_SIZE
import torch

S = DECODE_SEQ
EPS = M.rms_norm_eps
HEAD_DIM = M.index_head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    x = tensors['x'].float()
    compress_state = tensors['compress_state']
    compress_state_block_table = tensors['compress_state_block_table']
    wkv = tensors['wkv'].float()
    wgate = tensors['wgate'].float()
    ape = tensors['ape']
    norm_w = tensors['norm_w']
    cos = tensors['cos']
    sin = tensors['sin']
    hadamard = tensors['hadamard'].float()
    idx_kv_cache = tensors['idx_kv_cache']
    idx_kv_scale = tensors['idx_kv_scale']
    position_ids = tensors['position_ids'].to(torch.int64)
    idx_slot_mapping = tensors['idx_slot_mapping'].to(torch.int64)
    inner_state_slot_mapping = tensors['inner_state_slot_mapping'].to(torch.int64)
    tokens = x.shape[0]
    bsz = tokens // S
    position_ids = position_ids.view(bsz, S)
    idx_slot_mapping = idx_slot_mapping.view(bsz, S)
    inner_state_slot_mapping = inner_state_slot_mapping.view(bsz, S)
    (ratio, rd) = (COMPRESS_RATIO, ROPE_HEAD_DIM)
    kv_proj = x @ wkv.t()
    score_proj = x @ wgate.t()
    old_state = compress_state.clone()
    pooled = torch.zeros(tokens, HEAD_DIM, dtype=torch.float32, device=x.device)
    for b in range(bsz):
        first_pos = int(position_ids[b, 0].item())
        for s in range(S):
            token = b * S + s
            token_pos = int(position_ids[b, s].item())
            if (token_pos + 1) % ratio != 0:
                continue
            kv_rows = []
            score_rows = []
            for state_idx in range(STATE_LEN):
                logical_pos = token_pos - STATE_LEN + 1 + state_idx
                state_half = 0 if state_idx < ratio else HEAD_DIM
                if logical_pos < 0:
                    kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device))
                    score_rows.append(torch.full((HEAD_DIM,), float('-inf'), dtype=torch.float32, device=x.device))
                elif logical_pos < first_pos:
                    ring_row = logical_pos % STATE_LEN
                    (page_off, intra) = divmod(ring_row, COMPRESS_STATE_BLOCK_SIZE)
                    block = int(compress_state_block_table[b, page_off].item())
                    if block >= 0:
                        kv_rows.append(old_state[block, intra, state_half:state_half + HEAD_DIM])
                        score_rows.append(old_state[block, intra, OUT_DIM + state_half:OUT_DIM + state_half + HEAD_DIM])
                    else:
                        kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device))
                        score_rows.append(torch.full((HEAD_DIM,), float('-inf'), dtype=torch.float32, device=x.device))
                else:
                    overlay_token = b * S + logical_pos - first_pos
                    kv_rows.append(kv_proj[overlay_token, state_half:state_half + HEAD_DIM])
                    score_rows.append(score_proj[overlay_token, state_half:state_half + HEAD_DIM] + ape[logical_pos % ratio, state_half:state_half + HEAD_DIM])
            kvs = torch.stack(kv_rows, dim=0)
            scores = torch.stack(score_rows, dim=0)
            pooled[token] = (kvs * scores.softmax(dim=0)).sum(dim=0)
    for b in range(bsz):
        for s in range(S):
            token = b * S + s
            state_row = int(inner_state_slot_mapping[b, s].item())
            if state_row < 0:
                continue
            (block, intra) = divmod(state_row, COMPRESS_STATE_BLOCK_SIZE)
            token_pos = int(position_ids[b, s].item())
            compress_state[block, intra, :OUT_DIM] = kv_proj[token]
            compress_state[block, intra, OUT_DIM:] = score_proj[token] + ape[token_pos % ratio]
    tensors['compress_state'][:] = compress_state

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x
    for b in range(bsz):
        for s in range(S):
            token = b * S + s
            token_pos = int(position_ids[b, s].item())
            if (token_pos + 1) % ratio != 0:
                continue
            kv_b = rmsnorm(pooled[token:token + 1], norm_w)
            rope_normed = kv_b[..., -rd:]
            rope_swapped = rope_normed.reshape(1, -1, 2).flip(-1).flatten(-2)
            rope_rot = rope_normed * cos[token] + rope_swapped * sin[token]
            kv_b = torch.cat([kv_b[..., :-rd], rope_rot], dim=-1)
            kv_b = kv_b.to(torch.bfloat16).float() @ hadamard
            cache_row = int(idx_slot_mapping[b, s].item())
            if cache_row < 0:
                continue
            tensors['kv'][token:token + 1, :] = kv_b
            blk_id = cache_row // BLOCK_SIZE
            intra = cache_row % BLOCK_SIZE
            row_bf16 = kv_b[0].to(torch.bfloat16).float()
            amax = row_bf16.abs().amax().clamp_min(INT8_AMAX_EPS)
            scale_q = INT8_SCALE_MAX / amax
            idx_kv_cache[blk_id, intra, 0] = torch.round(row_bf16 * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
            idx_kv_scale[blk_id, intra, 0, 0] = 1.0 / scale_q
    tensors['idx_kv_cache'][:] = idx_kv_cache
    tensors['idx_kv_scale'][:] = idx_kv_scale
