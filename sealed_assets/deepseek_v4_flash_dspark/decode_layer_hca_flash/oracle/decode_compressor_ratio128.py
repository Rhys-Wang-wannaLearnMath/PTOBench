"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_hca_flash."""

from config import CASE_CONFIG as M
from config import COMPRESS_RATIO
from config import DECODE_SEQ
from config import C128_COMPRESSOR_BLOCK_SIZE
from config import BLOCK_SIZE
import torch

S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
COFF = 1
OUT_DIM = COFF * HEAD_DIM
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE

def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap).

    Operates on paged caches: compress_state (kv + score channels merged) and cmp_kv_cache,
    each addressed via the corresponding block_table.
    """
    x = tensors['x'].float()
    compress_state_block_table = tensors['compress_state_block_table']
    position_ids = tensors['position_ids'].to(torch.int64)
    cmp_slot_mapping = tensors['cmp_slot_mapping'].to(torch.int64)
    state_slot_mapping = tensors['state_slot_mapping'].to(torch.int64)
    compress_state = tensors['compress_state']

    def read_state_row(b, pos):
        logical_blk = pos // COMPRESS_STATE_BLOCK_SIZE
        intra = pos % COMPRESS_STATE_BLOCK_SIZE
        sblk = int(compress_state_block_table[b, logical_blk].item())
        if sblk < 0:
            return (torch.zeros(OUT_DIM, dtype=torch.float32, device=compress_state.device), torch.full((OUT_DIM,), float('-inf'), dtype=torch.float32, device=compress_state.device))
        return (compress_state[sblk, intra, :OUT_DIM], compress_state[sblk, intra, OUT_DIM:2 * OUT_DIM])

    def write_state_row(slot, kv_row, score_row):
        if slot < 0:
            return
        sblk = slot // COMPRESS_STATE_BLOCK_SIZE
        intra = slot % COMPRESS_STATE_BLOCK_SIZE
        compress_state[sblk, intra, :OUT_DIM] = kv_row
        compress_state[sblk, intra, OUT_DIM:2 * OUT_DIM] = score_row
    wkv = tensors['wkv'].float()
    wgate = tensors['wgate'].float()
    ape = tensors['ape']
    norm_w = tensors['norm_w']
    cos = tensors['cos']
    sin = tensors['sin']
    cmp_kv_cache = tensors['cmp_kv_cache']
    tokens = x.shape[0]
    bsz = tokens // S
    x = x.view(bsz, S, D)
    position_ids = position_ids.view(bsz, S)
    cmp_slot_mapping = cmp_slot_mapping.view(bsz, S)
    state_slot_mapping = state_slot_mapping.view(bsz, S)
    (ratio, rd) = (COMPRESS_RATIO, ROPE_HEAD_DIM)
    kv = x @ wkv.t()
    score = x @ wgate.t()
    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)
    for b in range(bsz):
        boundary_s = None
        for s in range(S):
            pos = int(position_ids[b, s].item())
            token_ape_row = pos % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            write_state_row(int(state_slot_mapping[b, s].item()), kv[b, s, :], score[b, s, :])
            if (pos + 1) % ratio == 0:
                boundary_s = s
        if boundary_s is not None:
            should_compress_rows[b] = True
            compress_pos = int(position_ids[b, boundary_s].item())
            kv_rows = []
            score_rows = []
            for pos in range(compress_pos - ratio + 1, compress_pos + 1):
                (kv_row, score_row) = read_state_row(b, pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            kv_state = torch.stack(kv_rows, dim=0).unsqueeze(0)
            score_state = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b:b + 1] = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
    tensors['compress_state'][:] = compress_state
    if not bool(should_compress_rows.any()):
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x
    for b in range(bsz):
        if not bool(should_compress_rows[b]):
            continue
        kv_b = rmsnorm(pooled[b:b + 1], norm_w)
        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        (x0, x1) = (x_pair[..., 0], x_pair[..., 1])
        (cos_v, sin_v) = (cos[b].view(-1), sin[b].view(-1))
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v
        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
        boundary_positions = torch.nonzero((position_ids[b, :S] + 1) % ratio == 0, as_tuple=False).flatten()
        if int(boundary_positions.numel()) == 0:
            continue
        boundary_s = int(boundary_positions[0].item())
        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if cmp_row >= 0:
            tensors['kv'][b * S:b * S + 1, :] = kv_b[0]
            cblk = cmp_row // BLOCK_SIZE
            intra_offset = cmp_row % BLOCK_SIZE
            cmp_kv_cache[cblk, intra_offset, 0] = kv_b[0, 0]
    tensors['cmp_kv_cache'][:] = cmp_kv_cache
