"""Independent CPU Oracle for qwen3_14b.qwen3_14b_decode_tq_single_layer."""

import math as _math
import torch
from config import (
    Q_HEAD_BATCH, Q_HEAD_PAD, BLOCK_SIZE, N_LEVELS, EPS, TQ_BITS,
    TQ_MAX_ITER, TQ_TOL, TQ_INIT_LO, TQ_INIT_HI, TQ_CDF_GUARD,
)

BATCH_TILE = 16
SCOPE1_K_CHUNK = 512

def solve_lloyd_max(d: int, bits: int, max_iter: int = TQ_MAX_ITER, tol: float = TQ_TOL):
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d).

    Uses fully vectorized operations:
    - Analytical Gaussian CDF via torch.erf for centroid updates
    - No per-element Python loops in the hot path

    Returns:
        centroids: sorted tensor of 2^bits optimal centroids
        boundaries: sorted tensor of 2^bits - 1 boundaries
    """
    import torch

    n_levels = 2 ** bits
    sigma = 1.0 / _math.sqrt(d)

    # Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
    lo, hi = TQ_INIT_LO, TQ_INIT_HI
    centroids = torch.linspace(lo, hi, n_levels + 2)[1:-1]  # n_levels points, excluding endpoints

    for _ in range(max_iter):
        # Step 1: boundaries = midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Step 2: update centroids via analytical Gaussian conditional mean.
        # For N(0, sigma^2) in interval [a, b]:
        #   E[X | a < X < b] = sigma * (phi(a/sigma) - phi(b/sigma))
        #                       / (Phi(b/sigma) - Phi(a/sigma))
        # where phi = standard normal PDF, Phi = CDF.
        # phi(x) = exp(-x^2/2) / sqrt(2pi), Phi(x) = 0.5*(1+erf(x/sqrt(2)))
        edges = torch.zeros(n_levels + 1)
        edges[0] = lo * 3
        edges[1:-1] = boundaries
        edges[-1] = hi * 3

        # Standardize edges
        z = edges / sigma

        # phi(z) = exp(-z^2/2) / sqrt(2*pi)
        neg_half_z2 = -0.5 * z * z
        phi_z = neg_half_z2.exp() * (1.0 / _math.sqrt(2.0 * _math.pi))

        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        cdf_z = 0.5 * (1.0 + torch.erf(z * (1.0 / _math.sqrt(2.0))))

        # Conditional means: sigma * (phi(z_a) - phi(z_b)) / (Phi(z_b) - Phi(z_a))
        phi_diff = phi_z[:-1] - phi_z[1:]   # phi(z_a) - phi(z_b), shape (n_levels,)
        cdf_diff = cdf_z[1:] - cdf_z[:-1]   # Phi(z_b) - Phi(z_a), shape (n_levels,)

        # Guard against tiny intervals
        valid = cdf_diff > TQ_CDF_GUARD
        new_centroids = torch.where(
            valid,
            sigma * phi_diff / cdf_diff.clamp(min=TQ_CDF_GUARD),
            centroids,
        )

        # Check convergence
        max_shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.clone(), boundaries.clone()

def golden_decode_fwd_tq(tensors):
    """PyTorch reference for the full-layer Qwen3-14B TQ decode program."""
    import math

    import torch


    hidden_states = tensors["hidden_states"].clone()
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    quant_k_cache = tensors["quant_k_cache"].clone()
    quant_v_cache = tensors["quant_v_cache"].clone()
    quant_k_scales = tensors["quant_k_scales"].clone()
    quant_v_scales = tensors["quant_v_scales"].clone()
    rot_matrices = tensors["rot_matrices"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_layers = input_rms_weight.shape[0]
    kv_hidden_size = wk.shape[1]
    num_kv_heads = kv_hidden_size // head_dim
    num_heads = hidden_size // head_dim
    intermediate_size = w_gate.shape[1]
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = EPS
    max_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    layer_cmp_cache_rows = batch * max_blocks_per_seq * num_kv_heads * BLOCK_SIZE

    # Solve Lloyd-Max codebook.
    lm_bits = TQ_BITS
    centroids, boundaries = solve_lloyd_max(head_dim, lm_bits)
    centroids = centroids.tolist()
    boundaries = boundaries.tolist()

    def _quantize_row(vec, rot_matrix):
        """Quantize a 1-D vector: L2 norm -> normalize -> rotate -> Lloyd-Max.

        L2 norm matches the kernel: l2 = sqrt(sum(x²) + eps), and normalize
        via recip + row_expand_mul equivalent.  Boundary search uses cumulative
        GE (rotated >= bval), matching pl.cmp cmp_type=5 (GE).
        """
        x_f = vec.float()
        l2 = torch.sqrt(x_f.pow(2).sum() + eps)  # != norm().clamp(min=eps)
        inv_norm = 1.0 / l2
        normed_bf16 = (x_f * inv_norm).to(torch.bfloat16)
        rotated = normed_bf16.float() @ rot_matrix.float()
        # Boundary search: cumulative GE count (== pl.cmp cmp_type=5).
        indices = torch.zeros(rotated.shape, dtype=torch.long)
        for bi, bval in enumerate(boundaries):
            indices += (rotated >= bval).long()
        # Pack two 4-bit indices per UINT8 (half/half layout), matching the kernel:
        # byte c = (idx[c+half] << 4) | idx[c].  Cast mirrors kernel INT32->FP16->UINT8.
        packed = (indices[half:] << 4) | indices[:half]              # [half] long
        return packed.to(torch.float16).to(torch.uint8), l2, rotated

    def _dequant_block(cache_rows_slice, scales_rows_slice, rot_mat):
        """Dequantize compressed cache rows: centroid lookup + renormalize + scale (rotated) + unrotate.

        Renorm uses rsqrt (== mul * rsqrt(sq+eps)), matching the kernel's
        qk_renorm / sv_dequant.  Scale is applied in the rotated domain before
        unrotate.  Index cast goes through UINT8→FP16→INT32.
        """
        # Unpack nibbles (half/half layout, matching the kernel): byte c -> idx[c]
        # (lo), idx[c+half] (hi).  Cast mirrors kernel UINT8->FP16->INT32.
        packed = cache_rows_slice.to(torch.float16).to(torch.int32)  # [rows, half]
        lo = packed & 0xF                                            # [rows, half]
        hi = packed >> 4                                             # [rows, half]
        indices = torch.cat([lo, hi], dim=-1)                        # [rows, head_dim]
        dec = torch.zeros(indices.shape, dtype=torch.float32)
        for ci in range(len(centroids)):
            dec += (indices == ci).float() * centroids[ci]
        # Renormalize to unit sphere (rsqrt == kernel's qk_renorm).
        dec_sq = dec.pow(2).sum(dim=-1, keepdim=True)
        dec = dec * torch.rsqrt(dec_sq + eps)
        # Rescale by stored L2 norms in the ROTATED domain (kernel order).
        dec = dec * scales_rows_slice.float()  # [rows, 1] BF16→FP32
        # Unrotate back to original space.
        return dec @ rot_mat.float().T

    hidden = hidden_states
    for layer_idx in range(num_layers):
        layer_hidden_base = layer_idx * hidden_size
        layer_inter_base = layer_idx * intermediate_size
        layer_cmp_base = layer_idx * layer_cmp_cache_rows
        rot_base = layer_idx * head_dim
        rot_matrix = rot_matrices[rot_base : rot_base + head_dim, :].float()  # [head_dim, head_dim]

        layer_wq = wq[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wk = wk[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wv = wv[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wo = wo[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_gate = w_gate[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_up = w_up[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_down = w_down[layer_inter_base : layer_inter_base + intermediate_size, :]

        # ── Scope 1: RMSNorm + Q/K/V projection ──
        q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
        k_proj = torch.zeros(batch, kv_hidden_size, dtype=torch.float32)
        v_proj = torch.zeros(batch, kv_hidden_size, dtype=torch.float32)

        for b0 in range(0, batch, BATCH_TILE):
            b_end = min(b0 + BATCH_TILE, batch)
            x_tile = hidden[b0:b_end, :].float()
            sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
            for k0 in range(0, hidden_size, SCOPE1_K_CHUNK):
                x_chunk = x_tile[:, k0 : k0 + SCOPE1_K_CHUNK]
                sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
            normed = (
                x_tile
                * torch.rsqrt(sq_sum / hidden_size + eps)
                * input_rms_weight[layer_idx : layer_idx + 1, :].float()
            ).bfloat16()
            q_proj[b0:b_end, :] = (normed.float() @ layer_wq.float()).float()
            k_proj[b0:b_end, :] = (normed.float() @ layer_wk.float()).float()
            v_proj[b0:b_end, :] = (normed.float() @ layer_wv.float()).float()

        # ── Q/K per-head norm ──
        for b_idx in range(batch):
            k_heads = k_proj[b_idx].view(num_kv_heads, head_dim)
            k_heads = (
                k_heads
                * torch.rsqrt(k_heads.pow(2).mean(dim=-1, keepdim=True) + eps)
                * k_norm_weight[layer_idx : layer_idx + 1, :].float()
            )
            k_proj[b_idx] = k_heads.reshape(-1)

            q_heads = q_proj[b_idx].view(num_heads, head_dim)
            q_heads = (
                q_heads
                * torch.rsqrt(q_heads.pow(2).mean(dim=-1, keepdim=True) + eps)
                * q_norm_weight[layer_idx : layer_idx + 1, :].float()
            )
            q_proj[b_idx] = q_heads.reshape(-1)

        # ── Scope 2: RoPE + KV quantize + Q RoPE + attention ──
        attn_out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
        for b_idx in range(batch):
            ctx_len = int(seq_lens[b_idx].item())
            pos = ctx_len - 1
            ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            cos_row = rope_cos[pos : pos + 1, :]
            sin_row = rope_sin[pos : pos + 1, :]
            cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
            sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

            slot = int(slot_mapping[b_idx].item())
            slot_block = slot // BLOCK_SIZE
            slot_offset = slot % BLOCK_SIZE

            # ── K: RoPE + L2 norm + normalize + rotate + quantize ──
            k_heads = k_proj[b_idx].view(num_kv_heads, head_dim)
            k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
            k_rope = torch.cat(
                [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
                dim=-1,
            )  # [num_kv_heads, head_dim]

            # Quantize K/V and collect L2 norms.
            k_l2_norms = []
            v_l2_norms = []
            for ki in range(num_kv_heads):
                k_vec = k_rope[ki]  # [head_dim]
                quant_cache_row = layer_cmp_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                k_idx_u8, k_l2, k_rot = _quantize_row(k_vec, rot_matrix)
                quant_k_cache[quant_cache_row, :] = k_idx_u8
                k_l2_norms.append(k_l2.item())

                # ── V: L2 norm + normalize + rotate + quantize (no RoPE) ──
                v_vec = v_proj[b_idx, ki * head_dim : (ki + 1) * head_dim]
                v_idx_u8, v_l2, _ = _quantize_row(v_vec, rot_matrix)
                quant_v_cache[quant_cache_row, :] = v_idx_u8
                v_l2_norms.append(v_l2.item())

            # Write K/V scales: [1] per (token, kvh) cache row.
            for ki in range(num_kv_heads):
                quant_cache_row = layer_cmp_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                quant_k_scales[quant_cache_row, 0] = k_l2_norms[ki]
                quant_v_scales[quant_cache_row, 0] = v_l2_norms[ki]

            # ── Q: RoPE ──
            q_heads = q_proj[b_idx].view(num_heads, head_dim)
            q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
            q_rope = torch.cat(
                [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
                dim=-1,
            )  # [num_heads, head_dim]

            # ── Attention: dequant from quant cache + QK/SV matmul ──
            attn_row_padded = torch.zeros(1, total_q_groups * Q_HEAD_PAD * head_dim, dtype=torch.bfloat16)
            for kvh in range(num_kv_heads):
                for qg in range(q_groups):
                    gi = kvh * q_groups + qg
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_grp = q_rope[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                    oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                    li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                    mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                    # All blocks are compressed; compute from ctx_len.
                    n_cmp_blks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

                    for sb in range(n_cmp_blks):
                        # Dequantize K for this block.
                        bt_idx = b_idx * max_blocks_per_seq + sb
                        pbid = int(block_table[bt_idx].item())
                        row_base = layer_cmp_base + (pbid * num_kv_heads + kvh) * BLOCK_SIZE

                        k_cache_block = quant_k_cache[row_base : row_base + BLOCK_SIZE, :]
                        k_scales_block = quant_k_scales[row_base : row_base + BLOCK_SIZE, 0:1]
                        k_dec = _dequant_block(k_cache_block, k_scales_block, rot_matrix)

                        # Dequantize V for this block.
                        v_cache_block = quant_v_cache[row_base : row_base + BLOCK_SIZE, :]
                        v_scales_block = quant_v_scales[row_base : row_base + BLOCK_SIZE, 0:1]
                        v_dec = _dequant_block(v_cache_block, v_scales_block, rot_matrix)

                        # QK matmul.
                        raw_scores = q_grp.float() @ k_dec.T  # [Q_HEAD_BATCH, BLOCK_SIZE]
                        s0 = sb * BLOCK_SIZE
                        valid_len = min(BLOCK_SIZE, ctx_len - s0)
                        if valid_len < BLOCK_SIZE:
                            raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                        scores = raw_scores * scale

                        # Softmax.
                        cur_mi = scores.max(dim=-1, keepdim=True).values
                        exp_scores = torch.exp(scores - cur_mi)
                        exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                        cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                        # SV matmul.
                        oi_tmp = exp_scores_bf16.float() @ v_dec

                        # Online softmax accumulation.
                        if sb == 0:
                            oi = oi_tmp
                            li = cur_li
                            mi = cur_mi
                        else:
                            mi_new = torch.maximum(mi, cur_mi)
                            alpha = torch.exp(mi - mi_new)
                            beta = torch.exp(cur_mi - mi_new)
                            li = alpha * li + beta * cur_li
                            oi = oi * alpha + oi_tmp * beta
                            mi = mi_new

                    ctx = oi / li  # [Q_HEAD_BATCH, head_dim]

                    # No unrotate needed: K/V are already in original space after dequant.
                    ctx_bf16 = ctx.to(torch.bfloat16)
                    ctx_flat_padded = torch.zeros(1, Q_HEAD_PAD * head_dim, dtype=torch.bfloat16)
                    ctx_flat_padded[:, : Q_HEAD_BATCH * head_dim] = ctx_bf16.reshape(1, -1)
                    attn_row_padded[
                        :,
                        gi * Q_HEAD_PAD * head_dim : (gi + 1) * Q_HEAD_PAD * head_dim,
                    ] = ctx_flat_padded

            # Write back attention output (strip Q_HEAD_PAD).
            attn_row = torch.zeros(1, hidden_size, dtype=torch.bfloat16)
            for kvh in range(num_kv_heads):
                for qg in range(q_groups):
                    gi = kvh * q_groups + qg
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    attn_row[
                        :,
                        q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim,
                    ] = attn_row_padded[
                        :,
                        gi * Q_HEAD_PAD * head_dim : gi * Q_HEAD_PAD * head_dim + Q_HEAD_BATCH * head_dim,
                    ]
            attn_out[b_idx : b_idx + 1, :] = attn_row

        # ── Scope 3: output projection + residual + post RMSNorm + MLP + final residual ──
        o_proj = attn_out.float() @ layer_wo.float()
        resid1 = o_proj + hidden.float()
        normed_bf16 = (
            resid1
            * torch.rsqrt(resid1.pow(2).mean(dim=-1, keepdim=True) + eps)
            * post_rms_weight[layer_idx : layer_idx + 1, :].float()
        ).bfloat16()
        gate = normed_bf16.float() @ layer_w_gate.float()
        up = normed_bf16.float() @ layer_w_up.float()
        mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
        down = mlp_bf16.float() @ layer_w_down.float()
        hidden = (down + resid1).bfloat16()

    # Final norm + LM head (matching the NPU kernel's rms_lm_head).
    final_norm_w = tensors["final_norm_weight"].float()          # [1, HIDDEN]
    lm_head_w = tensors["lm_head_weight"].float()               # [VOCAB, HIDDEN]
    hidden_f = hidden.float()
    variance = hidden_f.pow(2).mean(dim=-1, keepdim=True) + eps
    normed = hidden_f * torch.rsqrt(variance) * final_norm_w
    logits = normed @ lm_head_w.T                                # [batch, VOCAB]
    tensors["out"][:] = logits

def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors["out"] = torch.zeros(
        (inputs["hidden_states"].shape[0], inputs["lm_head_weight"].shape[0]),
        dtype=torch.float32,
    )
    golden_decode_fwd_tq(tensors)
    return {"out": tensors["out"].detach().cpu().contiguous()}
