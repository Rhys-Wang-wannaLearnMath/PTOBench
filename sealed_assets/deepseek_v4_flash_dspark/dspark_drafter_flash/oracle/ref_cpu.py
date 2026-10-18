# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from config import CASE_CONFIG as M, BLOCK_SIZE, D, DECODE_SEQ, DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_NOISE_TOKEN_ID, DSPARK_QUERY_PAD, DSPARK_QUERY_WIDTH, HC_MULT, HEAD_DIM, N_RANKS

def golden_dspark_drafter(tensors):
    import torch

    def rms_norm(hidden):
        hidden_fp32 = hidden.float()
        inv_rms = torch.rsqrt(hidden_fp32.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
        return (hidden_fp32 * inv_rms).to(torch.bfloat16)

    def project_kv(hidden):
        projected = hidden.float()[..., :HEAD_DIM]
        inv_rms = torch.rsqrt(projected.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
        return (projected * inv_rms).to(torch.bfloat16)

    def hc_coefficients(token_count):
        pre = torch.full((token_count, HC_MULT), 0.5 + M.hc_eps, dtype=torch.float32)
        post = torch.ones(token_count, HC_MULT, dtype=torch.float32)
        combine = torch.full((token_count, HC_MULT, HC_MULT), 0.25 + M.hc_eps, dtype=torch.float32)
        combine = combine / (combine.sum(-2, keepdim=True) + M.hc_eps)
        for _ in range(M.hc_sinkhorn_iters - 1):
            combine = combine / (combine.sum(-1, keepdim=True) + M.hc_eps)
            combine = combine / (combine.sum(-2, keepdim=True) + M.hc_eps)
        return pre, post, combine

    def hc_pre_zero_function(hidden):
        pre, post, combine = hc_coefficients(hidden.shape[0])
        mixed = hidden[:, 0] * pre[:, 0:1]
        for lane in range(1, HC_MULT):
            mixed = mixed + hidden[:, lane] * pre[:, lane : lane + 1]
        return mixed.to(torch.bfloat16), post, combine

    def hc_post_zero_output(residual, post, combine):
        output = torch.empty_like(residual)
        zero = torch.zeros(residual.shape[0], D, dtype=torch.float32)
        for output_lane in range(HC_MULT):
            row = zero * post[:, output_lane : output_lane + 1]
            for input_lane in range(HC_MULT):
                row = row + residual[:, input_lane] * combine[:, input_lane, output_lane : output_lane + 1]
            output[:, output_lane] = row
        return output

    tensors["head_hidden"].zero_()
    tensors["initial_hidden"].zero_()
    tensors["intermediate_hidden"].zero_()
    positions = tensors["anchor_positions"]
    tables = tensors["block_tables"]
    context_slots = tensors["context_slot_mapping"]
    for rank in range(N_RANKS):
        main_x = rms_norm(tensors["target_hidden"][rank, :, :D])
        query_ids = torch.zeros(DSPARK_MAX_BATCH * DSPARK_QUERY_PAD, dtype=torch.int64)
        for request in range(positions.shape[1]):
            row = request * DSPARK_QUERY_WIDTH
            query_ids[row] = tensors["anchor_token_ids"][rank, request]
            query_ids[row + 1 : row + DSPARK_QUERY_WIDTH] = DSPARK_NOISE_TOKEN_ID
        query_hidden = tensors["embedding_weight"][rank].index_select(0, query_ids)
        hidden = query_hidden.float().unsqueeze(1).expand(-1, HC_MULT, -1).contiguous()
        tensors["initial_hidden"][rank] = hidden

        for layer in range(DSPARK_DRAFT_LAYERS):
            cache = tensors["kv_caches"][rank, layer].view(-1, HEAD_DIM)
            layer_context_slots = context_slots[rank, layer]
            valid_context_slots = layer_context_slots[layer_context_slots >= 0].long()
            context_x = main_x[DECODE_SEQ - 1 :: DECODE_SEQ]
            cache[valid_context_slots] = project_kv(context_x[: valid_context_slots.numel()])
            query_slots = []
            for request in range(positions.shape[1]):
                anchor = int(positions[rank, request])
                for offset in range(1, DSPARK_QUERY_WIDTH + 1):
                    position = anchor + offset
                    physical_block = int(tables[rank, layer, request, position // BLOCK_SIZE])
                    query_slots.append(physical_block * BLOCK_SIZE + position % BLOCK_SIZE)
            mixed, post, combine = hc_pre_zero_function(hidden)
            query_normed = rms_norm(mixed[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH])
            active_tokens = positions.shape[1] * DSPARK_QUERY_WIDTH
            cache[torch.tensor(query_slots, dtype=torch.int64)] = project_kv(query_normed[:active_tokens])
            attention_hidden = hc_post_zero_output(
                hidden[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
                post[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
                combine[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
            )
            padded_attention = torch.zeros_like(hidden)
            padded_attention[: positions.shape[1] * DSPARK_QUERY_WIDTH] = attention_hidden[
                : positions.shape[1] * DSPARK_QUERY_WIDTH
            ]
            _, moe_post, moe_combine = hc_pre_zero_function(padded_attention)
            hidden = hc_post_zero_output(padded_attention, moe_post, moe_combine)
            tensors["intermediate_hidden"][rank, layer] = hidden

        active = hidden[: positions.shape[1] * DSPARK_QUERY_WIDTH]
        head_pre = torch.full((active.shape[0], HC_MULT), 0.5 + M.hc_eps, dtype=torch.float32)
        head = active[:, 0] * head_pre[:, 0:1]
        head = head + active[:, 1] * head_pre[:, 1:2]
        tail = active[:, 2] * head_pre[:, 2:3]
        tail = tail + active[:, 3] * head_pre[:, 3:4]
        head = (head + tail).to(torch.bfloat16)
        tensors["head_hidden"][rank] = head.view(positions.shape[1], DSPARK_QUERY_WIDTH, D)
