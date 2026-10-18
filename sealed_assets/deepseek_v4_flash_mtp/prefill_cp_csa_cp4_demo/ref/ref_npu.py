import pypto.language as pl
import pypto.language.distributed as pld
from config import BLOCK_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_STATE_PHYSICAL_BLOCKS, CASE_CONFIG as M, IDX_CACHE_MAX_BLOCKS, PREFILL_CMP_BLOCK_NUM, PREFILL_CMP_MAX_BLOCKS, PREFILL_IDX_BLOCK_NUM, PREFILL_ORI_MAX_BLOCKS
from hc_post import hc_post_prefill
from hc_pre import hc_pre
from prefill_compressor_ratio4 import HEAD_DIM, OUT_DIM as MAIN_OUT_DIM, _prefill_compressor_ratio4_with_completion
from prefill_cp_exchange import INNER_STATE_DIM, INNER_STATE_ROWS, MAIN_CACHE_ROWS, MAIN_STATE_DIM, MAIN_STATE_ROWS, META_DIM, RECORDS_PER_WINDOW, ROWS_PER_RANK, SCALE_TILE_COLS, STATE_META_DIM, STATE_RECORDS_PER_WINDOW, STATE_ROWS_PER_RANK, _prefill_cp_dual_tail_exchange_wave, _prefill_cp_csa_compact_finish_wave, _prefill_cp_csa_compact_transport_wave, _prefill_cp_sparse_stage
from prefill_cp_zigzag import CP_SIZE, CP_TAIL_WINDOW_ROWS, EPOCHS, NUM_SEGMENTS, TAIL_ROWS
from prefill_indexer import CP_INDEXER_SCORE_CAP, IDX_HEAD_DIM, IDX_N_HEADS, IDX_TOPK, _prefill_indexer_cp_score_topk
from prefill_indexer_compressor import OUT_DIM as INNER_OUT_DIM, _prefill_indexer_compressor_with_completion
from prefill_sparse_attn import PREFILL_SPARSE_PAD, ROPE_DIM, VALID_BLOCK_MASK_COLS, _prefill_sparse_attn_math
from qkv_proj_rope import materialize_rope_rows, qkv_proj_rope
from rmsnorm import rms_norm
D = M.hidden_size
H = M.num_attention_heads
HC_DIM = M.hc_dim
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
O_GROUPS = M.o_groups
O_GROUP_IN = H * M.head_dim // O_GROUPS
O_LORA = M.o_lora_rank
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
T = 128
COMPRESS_RATIO = 4
STATE_LEN = 8
WIN = M.sliding_window
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_SIZE
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2
MAX_COMPRESSED_ROWS_PER_TILE = T // COMPRESS_RATIO
MAX_COMPRESSED_ROWS_PER_SEGMENT = 2 * MAX_COMPRESSED_ROWS_PER_TILE
MAIN_STATE_BLOCK_SIZE = 4
INNER_STATE_BLOCK_SIZE = 4
MAIN_STATE_MAX_BLOCKS = (M.max_position_embeddings + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (M.max_position_embeddings + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * T
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD
MAX_COMPRESS_LEAVES = 1 + MAX_SEGMENT_TILES
MAIN_LEAF_CACHE_BLOCKS = PREFILL_CMP_BLOCK_NUM
MAIN_LEAF_CACHE_ROWS = MAIN_LEAF_CACHE_BLOCKS * BLOCK_SIZE
IDX_LEAF_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
IDX_LEAF_CACHE_ROWS = IDX_LEAF_CACHE_BLOCKS * BLOCK_SIZE
LOCAL_LEAVES = LOCAL_PARTS * MAX_COMPRESS_LEAVES
IDX_CACHE_ROWS = PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE
QK_ROPE_HALF = ROPE_HEAD_DIM // 2

@pl.jit.inline
def _cp_csa_compress_pack_part(effective_x: pl.Tensor[[MAX_COMPRESS_LEAVES * T, D], pl.BF16], leaf_positions: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT32], leaf_main_slots: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT64], leaf_idx_slots: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT64], leaf_main_state_slots: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT64], leaf_inner_state_slots: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT64], leaf_num_tokens: pl.Tensor[[MAX_COMPRESS_LEAVES], pl.INT32], main_state_workspace: pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], inner_state_workspace: pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], main_state_block_table: pl.Tensor[[MAIN_STATE_MAX_BLOCKS], pl.INT32], inner_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32], idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], segment: pl.Scalar[pl.INT32], segment_start: pl.Scalar[pl.INT32], segment_active_length: pl.Scalar[pl.INT32], final_segment: pl.Scalar[pl.INT32], input_ready: pl.Scalar[pl.TASK_ID]):
    """Build one owner part's compact payloads."""
    payload_rows = EPOCHS * MAX_COMPRESSED_ROWS_PER_SEGMENT
    state_payload_rows = EPOCHS * STATE_ROWS_PER_RANK
    main_cache = pl.create_tensor([PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    main_cache_initial_flat = pl.reshape(main_cache, [1, PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE * 1 * HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_main_cache') as main_cache_initial_tid:
        for main_cache_initial_block in pl.range(PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE * 1 * HEAD_DIM // 256):
            main_cache_initial_flat[0:1, main_cache_initial_block * 256:(main_cache_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE * 1 * HEAD_DIM % 256 != 0:
            for main_cache_initial_tail in pl.range(PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE * 1 * HEAD_DIM // 256 * 256, PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE * 1 * HEAD_DIM):
                pl.write(main_cache_initial_flat, [0, main_cache_initial_tail], pl.cast(0.0, pl.BF16))
    idx_cache = pl.create_tensor([PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], dtype=pl.INT8)
    idx_cache_initial_flat = pl.reshape(idx_cache, [1, PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * IDX_HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_idx_cache') as idx_cache_initial_tid:
        for idx_cache_initial_block in pl.range(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * IDX_HEAD_DIM // 256):
            idx_cache_initial_flat[0:1, idx_cache_initial_block * 256:(idx_cache_initial_block + 1) * 256] = pl.cast(pl.full([1, 256], dtype=pl.FP16, value=0.0), target_type=pl.INT8, mode='trunc')
        if PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * IDX_HEAD_DIM % 256 != 0:
            for idx_cache_initial_tail in pl.range(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * IDX_HEAD_DIM // 256 * 256, PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * IDX_HEAD_DIM):
                pl.write(idx_cache_initial_flat, [0, idx_cache_initial_tail], pl.cast(0, pl.INT8))
    idx_scale = pl.create_tensor([PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], dtype=pl.FP32)
    idx_scale_initial_flat = pl.reshape(idx_scale, [1, PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * 1])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_idx_scale') as idx_scale_initial_tid:
        for idx_scale_initial_block in pl.range(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * 1 // 256):
            idx_scale_initial_flat[0:1, idx_scale_initial_block * 256:(idx_scale_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        if PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * 1 % 256 != 0:
            for idx_scale_initial_tail in pl.range(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * 1 // 256 * 256, PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE * 1 * 1):
                pl.write(idx_scale_initial_flat, [0, idx_scale_initial_tail], pl.cast(0.0, pl.FP32))
    main_payload = pl.create_tensor([payload_rows, HEAD_DIM], dtype=pl.BF16)
    main_payload_initial_flat = pl.reshape(main_payload, [1, payload_rows * HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_main_payload') as main_payload_initial_tid:
        for main_payload_initial_block in pl.range(payload_rows * HEAD_DIM // 256):
            main_payload_initial_flat[0:1, main_payload_initial_block * 256:(main_payload_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if payload_rows * HEAD_DIM % 256 != 0:
            for main_payload_initial_tail in pl.range(payload_rows * HEAD_DIM // 256 * 256, payload_rows * HEAD_DIM):
                pl.write(main_payload_initial_flat, [0, main_payload_initial_tail], pl.cast(0.0, pl.BF16))
    idx_payload = pl.create_tensor([payload_rows, IDX_HEAD_DIM], dtype=pl.INT8)
    idx_payload_initial_flat = pl.reshape(idx_payload, [1, payload_rows * IDX_HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_idx_payload') as idx_payload_initial_tid:
        for idx_payload_initial_block in pl.range(payload_rows * IDX_HEAD_DIM // 256):
            idx_payload_initial_flat[0:1, idx_payload_initial_block * 256:(idx_payload_initial_block + 1) * 256] = pl.cast(pl.full([1, 256], dtype=pl.FP16, value=0.0), target_type=pl.INT8, mode='trunc')
        if payload_rows * IDX_HEAD_DIM % 256 != 0:
            for idx_payload_initial_tail in pl.range(payload_rows * IDX_HEAD_DIM // 256 * 256, payload_rows * IDX_HEAD_DIM):
                pl.write(idx_payload_initial_flat, [0, idx_payload_initial_tail], pl.cast(0, pl.INT8))
    idx_scale_payload = pl.create_tensor([payload_rows, SCALE_TILE_COLS], dtype=pl.FP32)
    idx_scale_payload_initial_flat = pl.reshape(idx_scale_payload, [1, payload_rows * SCALE_TILE_COLS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_idx_scale_payload') as idx_scale_payload_initial_tid:
        for idx_scale_payload_initial_block in pl.range(payload_rows * SCALE_TILE_COLS // 256):
            idx_scale_payload_initial_flat[0:1, idx_scale_payload_initial_block * 256:(idx_scale_payload_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        if payload_rows * SCALE_TILE_COLS % 256 != 0:
            for idx_scale_payload_initial_tail in pl.range(payload_rows * SCALE_TILE_COLS // 256 * 256, payload_rows * SCALE_TILE_COLS):
                pl.write(idx_scale_payload_initial_flat, [0, idx_scale_payload_initial_tail], pl.cast(0.0, pl.FP32))
    record_meta = pl.create_tensor([payload_rows, META_DIM], dtype=pl.INT32)
    record_meta_initial_flat = pl.reshape(record_meta, [payload_rows * META_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_record_meta') as record_meta_initial_tid:
        for record_meta_initial_element in pl.range(payload_rows * META_DIM):
            pl.write(record_meta_initial_flat, [record_meta_initial_element], pl.cast(-1, pl.INT32))
    main_state_payload = pl.create_tensor([state_payload_rows, MAIN_STATE_DIM], dtype=pl.FP32)
    main_state_payload_initial_flat = pl.reshape(main_state_payload, [1, state_payload_rows * MAIN_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_main_state_payload') as main_state_payload_initial_tid:
        for main_state_payload_initial_block in pl.range(state_payload_rows * MAIN_STATE_DIM // 256):
            main_state_payload_initial_flat[0:1, main_state_payload_initial_block * 256:(main_state_payload_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        if state_payload_rows * MAIN_STATE_DIM % 256 != 0:
            for main_state_payload_initial_tail in pl.range(state_payload_rows * MAIN_STATE_DIM // 256 * 256, state_payload_rows * MAIN_STATE_DIM):
                pl.write(main_state_payload_initial_flat, [0, main_state_payload_initial_tail], pl.cast(0.0, pl.FP32))
    inner_state_payload = pl.create_tensor([state_payload_rows, INNER_STATE_DIM], dtype=pl.FP32)
    inner_state_payload_initial_flat = pl.reshape(inner_state_payload, [1, state_payload_rows * INNER_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_inner_state_payload') as inner_state_payload_initial_tid:
        for inner_state_payload_initial_block in pl.range(state_payload_rows * INNER_STATE_DIM // 256):
            inner_state_payload_initial_flat[0:1, inner_state_payload_initial_block * 256:(inner_state_payload_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        if state_payload_rows * INNER_STATE_DIM % 256 != 0:
            for inner_state_payload_initial_tail in pl.range(state_payload_rows * INNER_STATE_DIM // 256 * 256, state_payload_rows * INNER_STATE_DIM):
                pl.write(inner_state_payload_initial_flat, [0, inner_state_payload_initial_tail], pl.cast(0.0, pl.FP32))
    main_state_meta = pl.create_tensor([state_payload_rows, STATE_META_DIM], dtype=pl.INT32)
    main_state_meta_initial_flat = pl.reshape(main_state_meta, [state_payload_rows * STATE_META_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_main_state_meta') as main_state_meta_initial_tid:
        for main_state_meta_initial_element in pl.range(state_payload_rows * STATE_META_DIM):
            pl.write(main_state_meta_initial_flat, [main_state_meta_initial_element], pl.cast(-1, pl.INT32))
    inner_state_meta = pl.create_tensor([state_payload_rows, STATE_META_DIM], dtype=pl.INT32)
    inner_state_meta_initial_flat = pl.reshape(inner_state_meta, [state_payload_rows * STATE_META_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_inner_state_meta') as inner_state_meta_initial_tid:
        for inner_state_meta_initial_element in pl.range(state_payload_rows * STATE_META_DIM):
            pl.write(inner_state_meta_initial_flat, [inner_state_meta_initial_element], pl.cast(-1, pl.INT32))
    pack_initialization_fence = pl.system.task_dummy(deps=[main_cache_initial_tid, idx_cache_initial_tid, idx_scale_initial_tid, main_payload_initial_tid, idx_payload_initial_tid, idx_scale_payload_initial_tid, record_meta_initial_tid, main_state_payload_initial_tid, inner_state_payload_initial_tid, main_state_meta_initial_tid, inner_state_meta_initial_tid, input_ready])
    main_state = main_state_workspace
    inner_state = inner_state_workspace
    for leaf in pl.unroll(MAX_COMPRESS_LEAVES):
        row0 = leaf * T
        x_leaf = pl.slice(effective_x, [T, D], [row0, 0])
        pos_leaf = pl.slice(leaf_positions, [T], [row0])
        main_map_leaf = pl.slice(leaf_main_slots, [T], [row0])
        idx_map_leaf = pl.slice(leaf_idx_slots, [T], [row0])
        main_state_map_leaf = pl.slice(leaf_main_state_slots, [T], [row0])
        inner_state_map_leaf = pl.slice(leaf_inner_state_slots, [T], [row0])
        active = pl.read(leaf_num_tokens, [leaf])
        active_eff = pl.max(active, 1)
        main_completion = pl.array.create(1, pl.TASK_ID)
        (main_cache_written, main_state_written) = _prefill_compressor_ratio4_with_completion(x_leaf, main_state, main_state_block_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, freqs_cos, freqs_sin, main_cache, pos_leaf, active_eff, main_map_leaf, main_state_map_leaf, main_completion, pack_initialization_fence)
        inner_completion = pl.array.create(1, pl.TASK_ID)
        (idx_cache_written, idx_scale_written, inner_state_written) = _prefill_indexer_compressor_with_completion(x_leaf, inner_state, inner_state_block_table, inner_wkv, inner_wgate, inner_ape, inner_norm_w, freqs_cos, freqs_sin, hadamard_idx, idx_cache, idx_scale, idx_block_table, pos_leaf, active_eff, idx_map_leaf, inner_state_map_leaf, inner_completion, pack_initialization_fence)
        main_state_next = pl.create_tensor([CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], dtype=pl.FP32)
        inner_state_next = pl.create_tensor([CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], dtype=pl.FP32)
        main_cache_flat = pl.reshape(main_cache_written, [MAIN_LEAF_CACHE_ROWS, HEAD_DIM])
        idx_cache_flat = pl.reshape(idx_cache_written, [IDX_LEAF_CACHE_ROWS, IDX_HEAD_DIM])
        idx_scale_flat = pl.reshape(idx_scale_written, [IDX_LEAF_CACHE_ROWS, 1])
        main_state_written_flat = pl.reshape(main_state_written, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
        inner_state_written_flat = pl.reshape(inner_state_written, [INNER_STATE_ROWS, INNER_STATE_DIM])
        main_state_next_flat = pl.reshape(main_state_next, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
        inner_state_next_flat = pl.reshape(inner_state_next, [INNER_STATE_ROWS, INNER_STATE_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_materialize_leaf', deps=[main_completion[0], inner_completion[0], pack_initialization_fence]):
            for state_row in pl.range(MAIN_STATE_ROWS):
                main_state_next_flat[state_row:state_row + 1, :] = main_state_written_flat[state_row:state_row + 1, :]
            for state_row in pl.range(INNER_STATE_ROWS):
                inner_state_next_flat[state_row:state_row + 1, :] = inner_state_written_flat[state_row:state_row + 1, :]
            if leaf > 0:
                segment_slot0 = segment_start // COMPRESS_RATIO
                for row in pl.range(T):
                    if row < active:
                        position = pl.read(pos_leaf, [row])
                        main_source_i64 = pl.read(main_map_leaf, [row])
                        idx_source_i64 = pl.read(idx_map_leaf, [row])
                        if main_source_i64 >= 0 and idx_source_i64 >= 0:
                            logical_slot = pl.cast((position + 1) // COMPRESS_RATIO - 1, pl.INT32)
                            record_row = logical_slot - segment_slot0
                            main_source = pl.cast(main_source_i64, pl.INDEX)
                            idx_source = pl.cast(idx_source_i64, pl.INDEX)
                            for epoch in pl.range(EPOCHS):
                                destination = epoch * MAX_COMPRESSED_ROWS_PER_SEGMENT + record_row
                                main_payload[destination:destination + 1, :] = main_cache_flat[main_source:main_source + 1, :]
                                idx_payload[destination:destination + 1, :] = idx_cache_flat[idx_source:idx_source + 1, :]
                                scale_value = pl.read(idx_scale_flat, [idx_source, 0])
                                for scale_col in pl.range(SCALE_TILE_COLS):
                                    pl.write(idx_scale_payload, [destination, scale_col], scale_value)
                                pl.write(record_meta, [destination, 0], pl.cast(1, pl.INT32))
                                pl.write(record_meta, [destination, 1], segment)
                                pl.write(record_meta, [destination, 2], position)
                                pl.write(record_meta, [destination, 3], pl.cast(record_row, pl.INT32))
                                pl.write(record_meta, [destination, 4], pl.cast(1, pl.INT32))
                                pl.write(record_meta, [destination, 5], logical_slot)
                                pl.write(record_meta, [destination, 6], pl.cast(1, pl.INT32))
                                pl.write(record_meta, [destination, 7], logical_slot)
        main_state = main_state_next
        inner_state = inner_state_next
    main_state_flat = pl.reshape(main_state, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
    inner_state_flat = pl.reshape(inner_state, [INNER_STATE_ROWS, INNER_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_pack_final_state'):
        if segment == final_segment:
            segment_end = segment_start + segment_active_length
            state_start = pl.max(segment_end - STATE_LEN, 0)
            for state_row in pl.range(STATE_ROWS_PER_RANK):
                position = pl.cast(state_start + state_row, pl.INT32)
                main_block = pl.read(main_state_block_table, [position // MAIN_STATE_BLOCK_SIZE])
                inner_block = pl.read(inner_state_block_table, [position // INNER_STATE_BLOCK_SIZE])
                if position < segment_end and main_block >= 0 and (inner_block >= 0):
                    main_source = pl.cast(main_block, pl.INDEX) * MAIN_STATE_BLOCK_SIZE + position % MAIN_STATE_BLOCK_SIZE
                    inner_source = pl.cast(inner_block, pl.INDEX) * INNER_STATE_BLOCK_SIZE + position % INNER_STATE_BLOCK_SIZE
                    for epoch in pl.range(EPOCHS):
                        destination = epoch * STATE_ROWS_PER_RANK + state_row
                        main_state_payload[destination:destination + 1, :] = main_state_flat[main_source:main_source + 1, :]
                        inner_state_payload[destination:destination + 1, :] = inner_state_flat[inner_source:inner_source + 1, :]
                        pl.write(main_state_meta, [destination, 0], pl.cast(1, pl.INT32))
                        pl.write(main_state_meta, [destination, 1], position)
                        pl.write(main_state_meta, [destination, 2], position)
                        pl.write(inner_state_meta, [destination, 0], pl.cast(1, pl.INT32))
                        pl.write(inner_state_meta, [destination, 1], position)
                        pl.write(inner_state_meta, [destination, 2], position)
    return (main_payload, idx_payload, idx_scale_payload, record_meta, main_state_payload, inner_state_payload, main_state_meta, inner_state_meta)

@pl.jit.inline
def prefill_cp_csa_core(x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], main_state_workspace0: pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], inner_state_workspace0: pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], main_state_workspace1: pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], inner_state_workspace1: pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], compress_state: pl.InOut[pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[MAIN_STATE_MAX_BLOCKS], pl.INT32], inner_compress_state: pl.InOut[pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32], owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32], predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32], query_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32], query_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32], overlay_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], pl.INT32], final_segment_t: pl.Tensor[[1], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[T], pl.INT32], final_win_row_src: pl.Tensor[[T], pl.INT32], final_slot_mapping: pl.Tensor[[T], pl.INT32], leaf_positions_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT32], leaf_main_slots_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_idx_slots_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_main_state_slots_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_inner_state_slots_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_num_tokens_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES], pl.INT32], effective_x_workspace: pl.Tensor[[LOCAL_LEAVES * T, D], pl.BF16], hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16], kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16], tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], main_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, HEAD_DIM], pl.BF16], idx_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, IDX_HEAD_DIM], pl.INT8], scale_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP32], record_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, META_DIM], pl.INT32], main_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], pl.FP32], main_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32], inner_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], pl.FP32], inner_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32], compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32]], my_rank: pl.Scalar[pl.INT32]):
    """Run inline CP-CSA attention for one rank."""
    main_state_workspace0_input_flat = pl.reshape(main_state_workspace0, [1, CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM])
    main_state_workspace0 = pl.create_tensor([CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], dtype=pl.FP32)
    main_state_workspace0_copy_flat = pl.reshape(main_state_workspace0, [1, CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_private_scratch_main_state_workspace0') as main_state_workspace0_copied_tid:
        for main_state_workspace0_copy_block in pl.range(CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM // 256):
            main_state_workspace0_copy_flat[0:1, main_state_workspace0_copy_block * 256:(main_state_workspace0_copy_block + 1) * 256] = main_state_workspace0_input_flat[0:1, main_state_workspace0_copy_block * 256:(main_state_workspace0_copy_block + 1) * 256]
        if CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM % 256 != 0:
            for main_state_workspace0_copy_tail in pl.range(CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM // 256 * 256, CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM):
                pl.write(main_state_workspace0_copy_flat, [0, main_state_workspace0_copy_tail], pl.read(main_state_workspace0_input_flat, [0, main_state_workspace0_copy_tail]))
    inner_state_workspace0_input_flat = pl.reshape(inner_state_workspace0, [1, CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM])
    inner_state_workspace0 = pl.create_tensor([CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], dtype=pl.FP32)
    inner_state_workspace0_copy_flat = pl.reshape(inner_state_workspace0, [1, CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_private_scratch_inner_state_workspace0') as inner_state_workspace0_copied_tid:
        for inner_state_workspace0_copy_block in pl.range(CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM // 256):
            inner_state_workspace0_copy_flat[0:1, inner_state_workspace0_copy_block * 256:(inner_state_workspace0_copy_block + 1) * 256] = inner_state_workspace0_input_flat[0:1, inner_state_workspace0_copy_block * 256:(inner_state_workspace0_copy_block + 1) * 256]
        if CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM % 256 != 0:
            for inner_state_workspace0_copy_tail in pl.range(CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM // 256 * 256, CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM):
                pl.write(inner_state_workspace0_copy_flat, [0, inner_state_workspace0_copy_tail], pl.read(inner_state_workspace0_input_flat, [0, inner_state_workspace0_copy_tail]))
    main_state_workspace1_input_flat = pl.reshape(main_state_workspace1, [1, CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM])
    main_state_workspace1 = pl.create_tensor([CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], dtype=pl.FP32)
    main_state_workspace1_copy_flat = pl.reshape(main_state_workspace1, [1, CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_private_scratch_main_state_workspace1') as main_state_workspace1_copied_tid:
        for main_state_workspace1_copy_block in pl.range(CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM // 256):
            main_state_workspace1_copy_flat[0:1, main_state_workspace1_copy_block * 256:(main_state_workspace1_copy_block + 1) * 256] = main_state_workspace1_input_flat[0:1, main_state_workspace1_copy_block * 256:(main_state_workspace1_copy_block + 1) * 256]
        if CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM % 256 != 0:
            for main_state_workspace1_copy_tail in pl.range(CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM // 256 * 256, CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE * MAIN_STATE_DIM):
                pl.write(main_state_workspace1_copy_flat, [0, main_state_workspace1_copy_tail], pl.read(main_state_workspace1_input_flat, [0, main_state_workspace1_copy_tail]))
    inner_state_workspace1_input_flat = pl.reshape(inner_state_workspace1, [1, CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM])
    inner_state_workspace1 = pl.create_tensor([CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], dtype=pl.FP32)
    inner_state_workspace1_copy_flat = pl.reshape(inner_state_workspace1, [1, CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_private_scratch_inner_state_workspace1') as inner_state_workspace1_copied_tid:
        for inner_state_workspace1_copy_block in pl.range(CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM // 256):
            inner_state_workspace1_copy_flat[0:1, inner_state_workspace1_copy_block * 256:(inner_state_workspace1_copy_block + 1) * 256] = inner_state_workspace1_input_flat[0:1, inner_state_workspace1_copy_block * 256:(inner_state_workspace1_copy_block + 1) * 256]
        if CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM % 256 != 0:
            for inner_state_workspace1_copy_tail in pl.range(CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM // 256 * 256, CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE * INNER_STATE_DIM):
                pl.write(inner_state_workspace1_copy_flat, [0, inner_state_workspace1_copy_tail], pl.read(inner_state_workspace1_input_flat, [0, inner_state_workspace1_copy_tail]))
    effective_x_workspace_input_flat = pl.reshape(effective_x_workspace, [1, LOCAL_LEAVES * T * D])
    effective_x_workspace = pl.create_tensor([LOCAL_LEAVES * T, D], dtype=pl.BF16)
    effective_x_workspace_copy_flat = pl.reshape(effective_x_workspace, [1, LOCAL_LEAVES * T * D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_private_scratch_effective_x_workspace') as effective_x_workspace_copied_tid:
        for effective_x_workspace_copy_block in pl.range(LOCAL_LEAVES * T * D // 256):
            effective_x_workspace_copy_flat[0:1, effective_x_workspace_copy_block * 256:(effective_x_workspace_copy_block + 1) * 256] = effective_x_workspace_input_flat[0:1, effective_x_workspace_copy_block * 256:(effective_x_workspace_copy_block + 1) * 256]
        if LOCAL_LEAVES * T * D % 256 != 0:
            for effective_x_workspace_copy_tail in pl.range(LOCAL_LEAVES * T * D // 256 * 256, LOCAL_LEAVES * T * D):
                pl.write(effective_x_workspace_copy_flat, [0, effective_x_workspace_copy_tail], pl.read(effective_x_workspace_input_flat, [0, effective_x_workspace_copy_tail]))
    root_scratch_ready = pl.system.task_dummy(deps=[main_state_workspace0_copied_tid, inner_state_workspace0_copied_tid, main_state_workspace1_copied_tid, inner_state_workspace1_copied_tid, effective_x_workspace_copied_tid])
    q = pl.create_tensor([LOCAL_ROWS, H, HEAD_DIM], dtype=pl.BF16)
    q_initial_flat = pl.reshape(q, [1, LOCAL_ROWS * H * HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_q'):
        for q_initial_block in pl.range(LOCAL_ROWS * H * HEAD_DIM // 256):
            q_initial_flat[0:1, q_initial_block * 256:(q_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if LOCAL_ROWS * H * HEAD_DIM % 256 != 0:
            for q_initial_tail in pl.range(LOCAL_ROWS * H * HEAD_DIM // 256 * 256, LOCAL_ROWS * H * HEAD_DIM):
                pl.write(q_initial_flat, [0, q_initial_tail], pl.cast(0.0, pl.BF16))
    post = pl.create_tensor([LOCAL_ROWS, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([LOCAL_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    rope_cos_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_cos_flat_initial_flat = pl.reshape(rope_cos_flat, [1, LOCAL_ROWS * ROPE_HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_rope_cos_flat'):
        for rope_cos_flat_initial_block in pl.range(LOCAL_ROWS * ROPE_HEAD_DIM // 256):
            rope_cos_flat_initial_flat[0:1, rope_cos_flat_initial_block * 256:(rope_cos_flat_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if LOCAL_ROWS * ROPE_HEAD_DIM % 256 != 0:
            for rope_cos_flat_initial_tail in pl.range(LOCAL_ROWS * ROPE_HEAD_DIM // 256 * 256, LOCAL_ROWS * ROPE_HEAD_DIM):
                pl.write(rope_cos_flat_initial_flat, [0, rope_cos_flat_initial_tail], pl.cast(0.0, pl.BF16))
    rope_sin_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_flat_initial_flat = pl.reshape(rope_sin_flat, [1, LOCAL_ROWS * ROPE_HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_rope_sin_flat'):
        for rope_sin_flat_initial_block in pl.range(LOCAL_ROWS * ROPE_HEAD_DIM // 256):
            rope_sin_flat_initial_flat[0:1, rope_sin_flat_initial_block * 256:(rope_sin_flat_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if LOCAL_ROWS * ROPE_HEAD_DIM % 256 != 0:
            for rope_sin_flat_initial_tail in pl.range(LOCAL_ROWS * ROPE_HEAD_DIM // 256 * 256, LOCAL_ROWS * ROPE_HEAD_DIM):
                pl.write(rope_sin_flat_initial_flat, [0, rope_sin_flat_initial_tail], pl.cast(0.0, pl.BF16))
    local_kv = pl.create_tensor([LOCAL_ROWS, HEAD_DIM], dtype=pl.BF16)
    local_kv_initial_flat = pl.reshape(local_kv, [1, LOCAL_ROWS * HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_local_kv'):
        for local_kv_initial_block in pl.range(LOCAL_ROWS * HEAD_DIM // 256):
            local_kv_initial_flat[0:1, local_kv_initial_block * 256:(local_kv_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if LOCAL_ROWS * HEAD_DIM % 256 != 0:
            for local_kv_initial_tail in pl.range(LOCAL_ROWS * HEAD_DIM // 256 * 256, LOCAL_ROWS * HEAD_DIM):
                pl.write(local_kv_initial_flat, [0, local_kv_initial_tail], pl.cast(0.0, pl.BF16))
    normed = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
    normed_initial_flat = pl.reshape(normed, [1, LOCAL_ROWS * D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_normed'):
        for normed_initial_block in pl.range(LOCAL_ROWS * D // 256):
            normed_initial_flat[0:1, normed_initial_block * 256:(normed_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if LOCAL_ROWS * D % 256 != 0:
            for normed_initial_tail in pl.range(LOCAL_ROWS * D // 256 * 256, LOCAL_ROWS * D):
                pl.write(normed_initial_flat, [0, normed_initial_tail], pl.cast(0.0, pl.BF16))
    mixed = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
    qr = pl.create_tensor([LOCAL_ROWS, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([LOCAL_ROWS, 1], dtype=pl.FP32)
    x_flat = pl.reshape(x_hc, [LOCAL_ROWS, HC_MULT, D])
    query_positions_flat = pl.reshape(query_positions, [LOCAL_ROWS])
    query_requests_flat = pl.reshape(query_requests, [LOCAL_ROWS])
    overlay_positions_flat = pl.reshape(overlay_positions, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    overlay_requests_flat = pl.reshape(overlay_requests, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    overlay_active_flat = pl.reshape(overlay_active_lengths, [NUM_LOCAL_TILES, OVERLAY_SOURCES])
    swa_indices_flat = pl.reshape(swa_indices, [LOCAL_ROWS, WIN])
    predecessor_segments_local = predecessor_segments
    segment_starts_local = segment_starts_t
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * T
        x_tile = pl.slice(x_flat, [T, HC_MULT, D], [row0, 0, 0])
        mixed_tile = pl.slice(mixed, [T, D], [row0, 0])
        post_tile = pl.slice(post, [T, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [T, HC_MULT * HC_MULT], [row0, 0])
        position_tile = pl.slice(query_positions_flat, [T], [row0])
        cos_tile = pl.slice(rope_cos_flat, [T, ROPE_HEAD_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [T, ROPE_HEAD_DIM], [row0, 0])
        normed_tile = pl.slice(normed, [T, D], [row0, 0])
        q_tile = pl.slice(q, [T, H, HEAD_DIM], [row0, 0, 0])
        kv_tile = pl.slice(local_kv, [T, HEAD_DIM], [row0, 0])
        qr_tile = pl.slice(qr, [T, Q_LORA], [row0, 0])
        qr_scale_tile = pl.slice(qr_scale, [T, 1], [row0, 0])
        hc_pre(x_tile, hc_attn_fn, hc_attn_scale, hc_attn_base, mixed_tile, post_tile, comb_tile)
        rms_tid = rms_norm(mixed_tile, attn_norm_w, normed_tile)
        late_dep = pl.system.task_dummy(deps=[rms_tid])
        active = pl.read(overlay_active_flat, [tile, 1])
        materialize_rope_rows(freqs_cos, freqs_sin, position_tile, active, cos_tile, sin_tile)
        qkv_proj_rope(normed_tile, wq_a, wq_b, wq_b_scale, wkv, cos_tile, sin_tile, gamma_cq, gamma_ckv, q_tile, kv_tile, qr_tile, qr_scale_tile, late_dep)
    local_hidden_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * T, D], dtype=pl.BF16)
    local_hidden_tail_initial_flat = pl.reshape(local_hidden_tail, [1, EPOCHS * LOCAL_PARTS * T * D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_local_hidden_tail'):
        for local_hidden_tail_initial_block in pl.range(EPOCHS * LOCAL_PARTS * T * D // 256):
            local_hidden_tail_initial_flat[0:1, local_hidden_tail_initial_block * 256:(local_hidden_tail_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if EPOCHS * LOCAL_PARTS * T * D % 256 != 0:
            for local_hidden_tail_initial_tail in pl.range(EPOCHS * LOCAL_PARTS * T * D // 256 * 256, EPOCHS * LOCAL_PARTS * T * D):
                pl.write(local_hidden_tail_initial_flat, [0, local_hidden_tail_initial_tail], pl.cast(0.0, pl.BF16))
    local_kv_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * T, HEAD_DIM], dtype=pl.BF16)
    local_kv_tail_initial_flat = pl.reshape(local_kv_tail, [1, EPOCHS * LOCAL_PARTS * T * HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_local_kv_tail'):
        for local_kv_tail_initial_block in pl.range(EPOCHS * LOCAL_PARTS * T * HEAD_DIM // 256):
            local_kv_tail_initial_flat[0:1, local_kv_tail_initial_block * 256:(local_kv_tail_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if EPOCHS * LOCAL_PARTS * T * HEAD_DIM % 256 != 0:
            for local_kv_tail_initial_tail in pl.range(EPOCHS * LOCAL_PARTS * T * HEAD_DIM // 256 * 256, EPOCHS * LOCAL_PARTS * T * HEAD_DIM):
                pl.write(local_kv_tail_initial_flat, [0, local_kv_tail_initial_tail], pl.cast(0.0, pl.BF16))
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_tail_assemble'):
        for part in pl.range(LOCAL_PARTS):
            active = pl.read(segment_active_lengths, [part])
            valid = pl.min(active, T)
            tail0 = pl.max(active - T, 0)
            for row in pl.range(T):
                if row < valid:
                    source = part * MAX_SEGMENT_TILES * T + tail0 + row
                    destination = part * T + row
                    local_hidden_tail[destination:destination + 1, :] = normed[source:source + 1, :]
                    local_kv_tail[destination:destination + 1, :] = local_kv[source:source + 1, :]
    logical_hidden = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    logical_kv = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_tail_exchange'):
        _prefill_cp_dual_tail_exchange_wave(local_hidden_tail, local_kv_tail, reverse_index, owner_rank_table, hidden_tail_window, kv_tail_window, tail_ready, tail_consumed, logical_hidden, logical_kv, my_rank, pl.cast(0, pl.INT32))
    effective_x = effective_x_workspace
    leaf_positions = pl.reshape(leaf_positions_input, [LOCAL_LEAVES * T])
    leaf_main_slots = pl.reshape(leaf_main_slots_input, [LOCAL_LEAVES * T])
    leaf_idx_slots = pl.reshape(leaf_idx_slots_input, [LOCAL_LEAVES * T])
    leaf_main_state_slots = pl.reshape(leaf_main_state_slots_input, [LOCAL_LEAVES * T])
    leaf_inner_state_slots = pl.reshape(leaf_inner_state_slots_input, [LOCAL_LEAVES * T])
    leaf_num_tokens = leaf_num_tokens_input
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_leaf_data_stage', deps=[root_scratch_ready]) as leaf_data_ready:
        for part in pl.range(LOCAL_PARTS):
            predecessor = pl.read(predecessor_segments_local, [part])
            predecessor_length = pl.cast(0, pl.INT32)
            if predecessor >= 0:
                predecessor_length = pl.cast(pl.min(pl.read(segment_lengths_t, [predecessor]), T), pl.INT32)
            seed_length = pl.read(leaf_num_tokens, [part, 0])
            seed_source0 = predecessor_length - seed_length
            for row in pl.range(T):
                leaf_row = part * MAX_COMPRESS_LEAVES * T + row
                if row < seed_length:
                    source = predecessor * T + seed_source0 + row
                    effective_x[leaf_row:leaf_row + 1, :] = logical_hidden[source:source + 1, :]
            for tile in pl.range(MAX_SEGMENT_TILES):
                leaf = 1 + tile
                active = pl.read(leaf_num_tokens, [part, leaf])
                for row in pl.range(T):
                    source = (part * MAX_SEGMENT_TILES + tile) * T + row
                    leaf_row = (part * MAX_COMPRESS_LEAVES + leaf) * T + row
                    if row < active:
                        effective_x[leaf_row:leaf_row + 1, :] = normed[source:source + 1, :]
    persistent_main_flat = pl.reshape(compress_state, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
    persistent_inner_flat = pl.reshape(inner_compress_state, [INNER_STATE_ROWS, INNER_STATE_DIM])
    scratch_main0_flat = pl.reshape(main_state_workspace0, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
    scratch_main1_flat = pl.reshape(main_state_workspace1, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
    scratch_inner0_flat = pl.reshape(inner_state_workspace0, [INNER_STATE_ROWS, INNER_STATE_DIM])
    scratch_inner1_flat = pl.reshape(inner_state_workspace1, [INNER_STATE_ROWS, INNER_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_seed_persistent_state', deps=[root_scratch_ready]) as persistent_seed_ready:
        if pl.read(owner_segments_t, [0]) == 0:
            for row in pl.range(MAIN_STATE_ROWS):
                scratch_main0_flat[row:row + 1, :] = persistent_main_flat[row:row + 1, :]
            for row in pl.range(INNER_STATE_ROWS):
                scratch_inner0_flat[row:row + 1, :] = persistent_inner_flat[row:row + 1, :]
        if pl.read(owner_segments_t, [1]) == 0:
            for row in pl.range(MAIN_STATE_ROWS):
                scratch_main1_flat[row:row + 1, :] = persistent_main_flat[row:row + 1, :]
            for row in pl.range(INNER_STATE_ROWS):
                scratch_inner1_flat[row:row + 1, :] = persistent_inner_flat[row:row + 1, :]
    compressor_input_ready = pl.system.task_dummy(deps=[leaf_data_ready, persistent_seed_ready])
    part_leaf_rows = MAX_COMPRESS_LEAVES * T
    leaf_num_tokens_flat = pl.reshape(leaf_num_tokens, [LOCAL_LEAVES])
    final_segment = pl.read(final_segment_t, [0])
    part0_x = pl.slice(effective_x, [part_leaf_rows, D], [0, 0])
    part0_positions = pl.slice(leaf_positions, [part_leaf_rows], [0])
    part0_main_slots = pl.slice(leaf_main_slots, [part_leaf_rows], [0])
    part0_idx_slots = pl.slice(leaf_idx_slots, [part_leaf_rows], [0])
    part0_main_state_slots = pl.slice(leaf_main_state_slots, [part_leaf_rows], [0])
    part0_inner_state_slots = pl.slice(leaf_inner_state_slots, [part_leaf_rows], [0])
    part0_num_tokens = pl.slice(leaf_num_tokens_flat, [MAX_COMPRESS_LEAVES], [0])
    part0_segment = pl.read(owner_segments_t, [0])
    part0_segment_start = pl.read(segment_starts_local, [pl.cast(part0_segment, pl.INDEX)])
    (part0_main_payload, part0_idx_payload, part0_idx_scale_payload, part0_record_meta, part0_main_state_payload, part0_inner_state_payload, part0_main_state_meta, part0_inner_state_meta) = _cp_csa_compress_pack_part(part0_x, part0_positions, part0_main_slots, part0_idx_slots, part0_main_state_slots, part0_inner_state_slots, part0_num_tokens, main_state_workspace0, inner_state_workspace0, compress_state_block_table, inner_compress_state_block_table, idx_block_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, inner_wkv, inner_wgate, inner_ape, inner_norm_w, freqs_cos, freqs_sin, hadamard_idx, part0_segment, part0_segment_start, pl.read(segment_active_lengths, [0]), final_segment, compressor_input_ready)
    part1_row0 = part_leaf_rows
    part1_x = pl.slice(effective_x, [part_leaf_rows, D], [part1_row0, 0])
    part1_positions = pl.slice(leaf_positions, [part_leaf_rows], [part1_row0])
    part1_main_slots = pl.slice(leaf_main_slots, [part_leaf_rows], [part1_row0])
    part1_idx_slots = pl.slice(leaf_idx_slots, [part_leaf_rows], [part1_row0])
    part1_main_state_slots = pl.slice(leaf_main_state_slots, [part_leaf_rows], [part1_row0])
    part1_inner_state_slots = pl.slice(leaf_inner_state_slots, [part_leaf_rows], [part1_row0])
    part1_num_tokens = pl.slice(leaf_num_tokens_flat, [MAX_COMPRESS_LEAVES], [MAX_COMPRESS_LEAVES])
    part1_segment = pl.read(owner_segments_t, [1])
    part1_segment_start = pl.read(segment_starts_local, [pl.cast(part1_segment, pl.INDEX)])
    (part1_main_payload, part1_idx_payload, part1_idx_scale_payload, part1_record_meta, part1_main_state_payload, part1_inner_state_payload, part1_main_state_meta, part1_inner_state_meta) = _cp_csa_compress_pack_part(part1_x, part1_positions, part1_main_slots, part1_idx_slots, part1_main_state_slots, part1_inner_state_slots, part1_num_tokens, main_state_workspace1, inner_state_workspace1, compress_state_block_table, inner_compress_state_block_table, idx_block_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, inner_wkv, inner_wgate, inner_ape, inner_norm_w, freqs_cos, freqs_sin, hadamard_idx, part1_segment, part1_segment_start, pl.read(segment_active_lengths, [1]), final_segment, compressor_input_ready)
    packed_main_payload = pl.create_tensor([EPOCHS * ROWS_PER_RANK, HEAD_DIM], dtype=pl.BF16)
    packed_main_payload_initial_flat = pl.reshape(packed_main_payload, [1, EPOCHS * ROWS_PER_RANK * HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_packed_main_payload') as packed_main_payload_initial_tid:
        for packed_main_payload_initial_block in pl.range(EPOCHS * ROWS_PER_RANK * HEAD_DIM // 256):
            packed_main_payload_initial_flat[0:1, packed_main_payload_initial_block * 256:(packed_main_payload_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        if EPOCHS * ROWS_PER_RANK * HEAD_DIM % 256 != 0:
            for packed_main_payload_initial_tail in pl.range(EPOCHS * ROWS_PER_RANK * HEAD_DIM // 256 * 256, EPOCHS * ROWS_PER_RANK * HEAD_DIM):
                pl.write(packed_main_payload_initial_flat, [0, packed_main_payload_initial_tail], pl.cast(0.0, pl.BF16))
    packed_idx_payload = pl.create_tensor([EPOCHS * ROWS_PER_RANK, IDX_HEAD_DIM], dtype=pl.INT8)
    packed_idx_payload_initial_flat = pl.reshape(packed_idx_payload, [1, EPOCHS * ROWS_PER_RANK * IDX_HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_packed_idx_payload') as packed_idx_payload_initial_tid:
        for packed_idx_payload_initial_block in pl.range(EPOCHS * ROWS_PER_RANK * IDX_HEAD_DIM // 256):
            packed_idx_payload_initial_flat[0:1, packed_idx_payload_initial_block * 256:(packed_idx_payload_initial_block + 1) * 256] = pl.cast(pl.full([1, 256], dtype=pl.FP16, value=0.0), target_type=pl.INT8, mode='trunc')
        if EPOCHS * ROWS_PER_RANK * IDX_HEAD_DIM % 256 != 0:
            for packed_idx_payload_initial_tail in pl.range(EPOCHS * ROWS_PER_RANK * IDX_HEAD_DIM // 256 * 256, EPOCHS * ROWS_PER_RANK * IDX_HEAD_DIM):
                pl.write(packed_idx_payload_initial_flat, [0, packed_idx_payload_initial_tail], pl.cast(0, pl.INT8))
    packed_idx_scale_payload = pl.create_tensor([EPOCHS * ROWS_PER_RANK, SCALE_TILE_COLS], dtype=pl.FP32)
    packed_idx_scale_payload_initial_flat = pl.reshape(packed_idx_scale_payload, [1, EPOCHS * ROWS_PER_RANK * SCALE_TILE_COLS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_packed_idx_scale_payload') as packed_idx_scale_payload_initial_tid:
        for packed_idx_scale_payload_initial_block in pl.range(EPOCHS * ROWS_PER_RANK * SCALE_TILE_COLS // 256):
            packed_idx_scale_payload_initial_flat[0:1, packed_idx_scale_payload_initial_block * 256:(packed_idx_scale_payload_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        if EPOCHS * ROWS_PER_RANK * SCALE_TILE_COLS % 256 != 0:
            for packed_idx_scale_payload_initial_tail in pl.range(EPOCHS * ROWS_PER_RANK * SCALE_TILE_COLS // 256 * 256, EPOCHS * ROWS_PER_RANK * SCALE_TILE_COLS):
                pl.write(packed_idx_scale_payload_initial_flat, [0, packed_idx_scale_payload_initial_tail], pl.cast(0.0, pl.FP32))
    packed_record_meta = pl.create_tensor([EPOCHS * ROWS_PER_RANK, META_DIM], dtype=pl.INT32)
    packed_record_meta_initial_flat = pl.reshape(packed_record_meta, [EPOCHS * ROWS_PER_RANK * META_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_packed_record_meta') as packed_record_meta_initial_tid:
        for packed_record_meta_initial_element in pl.range(EPOCHS * ROWS_PER_RANK * META_DIM):
            pl.write(packed_record_meta_initial_flat, [packed_record_meta_initial_element], pl.cast(-1, pl.INT32))
    packed_main_state_payload = pl.create_tensor([EPOCHS * STATE_ROWS_PER_RANK, MAIN_STATE_DIM], dtype=pl.FP32)
    packed_main_state_payload_initial_flat = pl.reshape(packed_main_state_payload, [1, EPOCHS * STATE_ROWS_PER_RANK * MAIN_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_packed_main_state_payload') as packed_main_state_payload_initial_tid:
        for packed_main_state_payload_initial_block in pl.range(EPOCHS * STATE_ROWS_PER_RANK * MAIN_STATE_DIM // 256):
            packed_main_state_payload_initial_flat[0:1, packed_main_state_payload_initial_block * 256:(packed_main_state_payload_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        if EPOCHS * STATE_ROWS_PER_RANK * MAIN_STATE_DIM % 256 != 0:
            for packed_main_state_payload_initial_tail in pl.range(EPOCHS * STATE_ROWS_PER_RANK * MAIN_STATE_DIM // 256 * 256, EPOCHS * STATE_ROWS_PER_RANK * MAIN_STATE_DIM):
                pl.write(packed_main_state_payload_initial_flat, [0, packed_main_state_payload_initial_tail], pl.cast(0.0, pl.FP32))
    packed_inner_state_payload = pl.create_tensor([EPOCHS * STATE_ROWS_PER_RANK, INNER_STATE_DIM], dtype=pl.FP32)
    packed_inner_state_payload_initial_flat = pl.reshape(packed_inner_state_payload, [1, EPOCHS * STATE_ROWS_PER_RANK * INNER_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_packed_inner_state_payload') as packed_inner_state_payload_initial_tid:
        for packed_inner_state_payload_initial_block in pl.range(EPOCHS * STATE_ROWS_PER_RANK * INNER_STATE_DIM // 256):
            packed_inner_state_payload_initial_flat[0:1, packed_inner_state_payload_initial_block * 256:(packed_inner_state_payload_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        if EPOCHS * STATE_ROWS_PER_RANK * INNER_STATE_DIM % 256 != 0:
            for packed_inner_state_payload_initial_tail in pl.range(EPOCHS * STATE_ROWS_PER_RANK * INNER_STATE_DIM // 256 * 256, EPOCHS * STATE_ROWS_PER_RANK * INNER_STATE_DIM):
                pl.write(packed_inner_state_payload_initial_flat, [0, packed_inner_state_payload_initial_tail], pl.cast(0.0, pl.FP32))
    packed_main_state_meta = pl.create_tensor([EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM], dtype=pl.INT32)
    packed_main_state_meta_initial_flat = pl.reshape(packed_main_state_meta, [EPOCHS * STATE_ROWS_PER_RANK * STATE_META_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_packed_main_state_meta') as packed_main_state_meta_initial_tid:
        for packed_main_state_meta_initial_element in pl.range(EPOCHS * STATE_ROWS_PER_RANK * STATE_META_DIM):
            pl.write(packed_main_state_meta_initial_flat, [packed_main_state_meta_initial_element], pl.cast(-1, pl.INT32))
    packed_inner_state_meta = pl.create_tensor([EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM], dtype=pl.INT32)
    packed_inner_state_meta_initial_flat = pl.reshape(packed_inner_state_meta, [EPOCHS * STATE_ROWS_PER_RANK * STATE_META_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_packed_inner_state_meta') as packed_inner_state_meta_initial_tid:
        for packed_inner_state_meta_initial_element in pl.range(EPOCHS * STATE_ROWS_PER_RANK * STATE_META_DIM):
            pl.write(packed_inner_state_meta_initial_flat, [packed_inner_state_meta_initial_element], pl.cast(-1, pl.INT32))
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_merge_part_payloads', deps=[packed_main_payload_initial_tid, packed_idx_payload_initial_tid, packed_idx_scale_payload_initial_tid, packed_record_meta_initial_tid, packed_main_state_payload_initial_tid, packed_inner_state_payload_initial_tid, packed_main_state_meta_initial_tid, packed_inner_state_meta_initial_tid]):
        for epoch in pl.range(EPOCHS):
            source_row0 = epoch * MAX_COMPRESSED_ROWS_PER_SEGMENT
            destination_epoch0 = epoch * ROWS_PER_RANK
            destination_part1 = destination_epoch0 + MAX_COMPRESSED_ROWS_PER_SEGMENT
            for row in pl.range(MAX_COMPRESSED_ROWS_PER_SEGMENT):
                source = source_row0 + row
                destination0 = destination_epoch0 + row
                destination1 = destination_part1 + row
                packed_main_payload[destination0:destination0 + 1, :] = part0_main_payload[source:source + 1, :]
                packed_main_payload[destination1:destination1 + 1, :] = part1_main_payload[source:source + 1, :]
                packed_idx_payload[destination0:destination0 + 1, :] = part0_idx_payload[source:source + 1, :]
                packed_idx_payload[destination1:destination1 + 1, :] = part1_idx_payload[source:source + 1, :]
                packed_idx_scale_payload[destination0:destination0 + 1, :] = part0_idx_scale_payload[source:source + 1, :]
                packed_idx_scale_payload[destination1:destination1 + 1, :] = part1_idx_scale_payload[source:source + 1, :]
                for col in pl.range(META_DIM):
                    pl.write(packed_record_meta, [destination0, col], pl.read(part0_record_meta, [source, col]))
                    pl.write(packed_record_meta, [destination1, col], pl.read(part1_record_meta, [source, col]))
            state_source0 = epoch * STATE_ROWS_PER_RANK
            for row in pl.range(STATE_ROWS_PER_RANK):
                state_source = state_source0 + row
                state_destination = state_source
                if pl.read(owner_segments_t, [0]) == final_segment:
                    packed_main_state_payload[state_destination:state_destination + 1, :] = part0_main_state_payload[state_source:state_source + 1, :]
                    packed_inner_state_payload[state_destination:state_destination + 1, :] = part0_inner_state_payload[state_source:state_source + 1, :]
                    for col in pl.range(STATE_META_DIM):
                        pl.write(packed_main_state_meta, [state_destination, col], pl.read(part0_main_state_meta, [state_source, col]))
                        pl.write(packed_inner_state_meta, [state_destination, col], pl.read(part0_inner_state_meta, [state_source, col]))
                if pl.read(owner_segments_t, [1]) == final_segment:
                    packed_main_state_payload[state_destination:state_destination + 1, :] = part1_main_state_payload[state_source:state_source + 1, :]
                    packed_inner_state_payload[state_destination:state_destination + 1, :] = part1_inner_state_payload[state_source:state_source + 1, :]
                    for col in pl.range(STATE_META_DIM):
                        pl.write(packed_main_state_meta, [state_destination, col], pl.read(part1_main_state_meta, [state_source, col]))
                        pl.write(packed_inner_state_meta, [state_destination, col], pl.read(part1_inner_state_meta, [state_source, col]))
    cmp_flat = pl.reshape(cmp_kv, [MAIN_CACHE_ROWS, HEAD_DIM])
    idx_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_ROWS, IDX_HEAD_DIM])
    idx_scale_flat = pl.reshape(idx_kv_scale, [IDX_CACHE_ROWS, 1])
    main_state_flat = pl.reshape(compress_state, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
    inner_state_flat = pl.reshape(inner_compress_state, [INNER_STATE_ROWS, INNER_STATE_DIM])
    for epoch in pl.range(EPOCHS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_compact_transport'):
            _prefill_cp_csa_compact_transport_wave(packed_main_payload, packed_idx_payload, packed_idx_scale_payload, packed_record_meta, packed_main_state_payload, packed_inner_state_payload, packed_main_state_meta, packed_inner_state_meta, main_window, idx_window, scale_window, record_window, main_state_window, main_state_meta_window, inner_state_window, inner_state_meta_window, compact_ready, compact_consumed, my_rank, pl.cast(epoch, pl.INT32))
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_receiver_commit'):
            for source_rank in pl.range(CP_SIZE):
                source_row = source_rank * ROWS_PER_RANK
                source_state_row = source_rank * STATE_ROWS_PER_RANK
                for row in pl.range(ROWS_PER_RANK):
                    meta_row = source_row + row
                    record_valid = pl.read(record_window, [meta_row, 0])
                    logical_segment = pl.read(record_window, [meta_row, 1])
                    boundary = pl.read(record_window, [meta_row, 2])
                    if record_valid > 0 and logical_segment >= 0 and (boundary >= 0):
                        main_valid = pl.read(record_window, [meta_row, 4])
                        main_slot = pl.read(record_window, [meta_row, 5])
                        if main_valid > 0 and main_slot >= 0:
                            logical_block = main_slot // BLOCK_SIZE
                            if logical_block < PREFILL_CMP_MAX_BLOCKS:
                                physical_block = pl.read(cmp_block_table, [logical_block])
                                if physical_block >= 0:
                                    destination = pl.cast(physical_block, pl.INDEX) * BLOCK_SIZE + main_slot % BLOCK_SIZE
                                    received_main_tile = main_window[meta_row:meta_row + 1, 0:HEAD_DIM]
                                    cmp_flat[destination:destination + 1, 0:HEAD_DIM] = received_main_tile
                        idx_valid = pl.read(record_window, [meta_row, 6])
                        idx_slot = pl.read(record_window, [meta_row, 7])
                        if idx_valid > 0 and idx_slot >= 0:
                            logical_block = idx_slot // BLOCK_SIZE
                            if logical_block < IDX_CACHE_MAX_BLOCKS:
                                physical_block = pl.read(idx_block_table, [logical_block])
                                if physical_block >= 0:
                                    destination = pl.cast(physical_block, pl.INDEX) * BLOCK_SIZE + idx_slot % BLOCK_SIZE
                                    received_idx_tile = idx_window[meta_row:meta_row + 1, 0:IDX_HEAD_DIM]
                                    idx_flat[destination:destination + 1, 0:IDX_HEAD_DIM] = received_idx_tile
                                    pl.write(idx_scale_flat, [destination, 0], pl.read(scale_window, [meta_row, 0]))
                for state_row in pl.range(STATE_ROWS_PER_RANK):
                    meta_row = source_state_row + state_row
                    main_valid = pl.read(main_state_meta_window, [meta_row, 0])
                    main_position = pl.read(main_state_meta_window, [meta_row, 2])
                    if main_valid > 0 and main_position >= 0:
                        logical_block = main_position // MAIN_STATE_BLOCK_SIZE
                        if logical_block < MAIN_STATE_MAX_BLOCKS:
                            physical_block = pl.read(compress_state_block_table, [logical_block])
                            if physical_block >= 0:
                                destination = pl.cast(physical_block, pl.INDEX) * MAIN_STATE_BLOCK_SIZE + main_position % MAIN_STATE_BLOCK_SIZE
                                received_main_state = main_state_window[meta_row:meta_row + 1, 0:MAIN_STATE_DIM]
                                main_state_flat[destination:destination + 1, 0:MAIN_STATE_DIM] = received_main_state
                    inner_valid = pl.read(inner_state_meta_window, [meta_row, 0])
                    inner_position = pl.read(inner_state_meta_window, [meta_row, 2])
                    if inner_valid > 0 and inner_position >= 0:
                        logical_block = inner_position // INNER_STATE_BLOCK_SIZE
                        if logical_block < INNER_STATE_MAX_BLOCKS:
                            physical_block = pl.read(inner_compress_state_block_table, [logical_block])
                            if physical_block >= 0:
                                destination = pl.cast(physical_block, pl.INDEX) * INNER_STATE_BLOCK_SIZE + inner_position % INNER_STATE_BLOCK_SIZE
                                received_inner_state = inner_state_window[meta_row:meta_row + 1, 0:INNER_STATE_DIM]
                                inner_destination_end = destination + 1
                                inner_state_flat[destination:inner_destination_end, :] = received_inner_state
            _prefill_cp_csa_compact_finish_wave(compact_consumed, my_rank)
    cmp_indices = pl.create_tensor([LOCAL_ROWS, IDX_TOPK], dtype=pl.INT32)
    cmp_indices_initial_flat = pl.reshape(cmp_indices, [1, LOCAL_ROWS * IDX_TOPK])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_cmp_indices'):
        for cmp_indices_initial_block in pl.range(LOCAL_ROWS * IDX_TOPK // 256):
            cmp_indices_initial_flat[0:1, cmp_indices_initial_block * 256:(cmp_indices_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.INT32, value=-1)
        if LOCAL_ROWS * IDX_TOPK % 256 != 0:
            for cmp_indices_initial_tail in pl.range(LOCAL_ROWS * IDX_TOPK // 256 * 256, LOCAL_ROWS * IDX_TOPK):
                pl.write(cmp_indices_initial_flat, [0, cmp_indices_initial_tail], pl.cast(-1, pl.INT32))
    for tile in pl.range(NUM_LOCAL_TILES):
        active = pl.read(overlay_active_flat, [tile, 1])
        if active > 0:
            row0 = tile * T
            index_x_tile = pl.slice(normed, [T, D], [row0, 0])
            qr_tile = pl.slice(qr, [T, Q_LORA], [row0, 0])
            qr_scale_tile = pl.slice(qr_scale, [T, 1], [row0, 0])
            pos_tile = pl.slice(query_positions_flat, [T], [row0])
            score = pl.create_tensor([T, CP_INDEXER_SCORE_CAP], dtype=pl.FP32)
            topk_tile = pl.slice(cmp_indices, [T, IDX_TOPK], [row0, 0])
            idx_cos = pl.create_tensor([T, QK_ROPE_HALF], dtype=pl.FP32)
            idx_sin = pl.create_tensor([T, QK_ROPE_HALF], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_indexer_rope_rows'):
                idx_cos[0:T, :] = pl.cast(rope_cos_flat[row0:row0 + T, 0:QK_ROPE_HALF], target_type=pl.FP32)
                idx_sin[0:T, :] = pl.cast(rope_sin_flat[row0:row0 + T, 0:QK_ROPE_HALF], target_type=pl.FP32)
            (score, topk_tile) = _prefill_indexer_cp_score_topk(index_x_tile, qr_tile, qr_scale_tile, idx_wq_b, idx_wq_b_scale, idx_weights_proj, idx_cos, idx_sin, hadamard_idx, idx_kv_cache, idx_kv_scale, idx_block_table, pos_tile, active, score, topk_tile)
            cmp_indices[row0:row0 + T, 0:IDX_TOPK] = topk_tile
    cache_flat = pl.reshape(kv_cache, [ORI_CACHE_ROWS, HEAD_DIM])
    sparse_kv = pl.create_tensor([LOCAL_SPARSE_ROWS, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([LOCAL_ROWS, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    sparse_bias_initial_flat = pl.reshape(sparse_bias, [1, LOCAL_ROWS * PREFILL_SPARSE_PAD])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_sparse_bias'):
        for sparse_bias_initial_block in pl.range(LOCAL_ROWS * PREFILL_SPARSE_PAD // 256):
            sparse_bias_initial_flat[0:1, sparse_bias_initial_block * 256:(sparse_bias_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=-3e+38)
        if LOCAL_ROWS * PREFILL_SPARSE_PAD % 256 != 0:
            for sparse_bias_initial_tail in pl.range(LOCAL_ROWS * PREFILL_SPARSE_PAD // 256 * 256, LOCAL_ROWS * PREFILL_SPARSE_PAD):
                pl.write(sparse_bias_initial_flat, [0, sparse_bias_initial_tail], pl.cast(-3e+38, pl.FP32))
    valid_mask = pl.create_tensor([LOCAL_ROWS, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    valid_mask_initial_flat = pl.reshape(valid_mask, [1, LOCAL_ROWS * VALID_BLOCK_MASK_COLS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_valid_mask'):
        for valid_mask_initial_block in pl.range(LOCAL_ROWS * VALID_BLOCK_MASK_COLS // 256):
            valid_mask_initial_flat[0:1, valid_mask_initial_block * 256:(valid_mask_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.INT32, value=0)
        if LOCAL_ROWS * VALID_BLOCK_MASK_COLS % 256 != 0:
            for valid_mask_initial_tail in pl.range(LOCAL_ROWS * VALID_BLOCK_MASK_COLS // 256 * 256, LOCAL_ROWS * VALID_BLOCK_MASK_COLS):
                pl.write(valid_mask_initial_flat, [0, valid_mask_initial_tail], pl.cast(0, pl.INT32))
    _prefill_cp_sparse_stage(cache_flat, local_kv, logical_kv, cmp_flat, cmp_block_table, query_positions_flat, query_requests_flat, overlay_positions_flat, overlay_requests_flat, predecessor_segments_local, segment_starts_local, swa_indices_flat, cmp_indices, sparse_kv, sparse_bias, valid_mask, overlay_active_flat)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_raw_commit'):
        for row in pl.range(T):
            raw_segment = pl.read(final_win_seg_src, [row])
            raw_source_row = pl.read(final_win_row_src, [row])
            raw_destination = pl.read(final_slot_mapping, [row])
            if raw_segment >= 0 and raw_source_row >= 0 and (raw_destination >= 0):
                raw_source = raw_segment * T + raw_source_row
                cache_flat[raw_destination:raw_destination + 1, :] = logical_kv[raw_source:raw_source + 1, :]
    x_out_flat = pl.reshape(x_out, [LOCAL_ROWS, HC_MULT, D])
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * T
        sparse_row0 = row0 * PREFILL_SPARSE_PAD
        q_tile = pl.slice(q, [T, H, HEAD_DIM], [row0, 0, 0])
        sparse_tile = pl.slice(sparse_kv, [T * PREFILL_SPARSE_PAD, HEAD_DIM], [sparse_row0, 0])
        bias_tile = pl.slice(sparse_bias, [T, PREFILL_SPARSE_PAD], [row0, 0])
        mask_tile = pl.slice(valid_mask, [T, VALID_BLOCK_MASK_COLS], [row0, 0])
        cos_tile = pl.slice(rope_cos_flat, [T, ROPE_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [T, ROPE_DIM], [row0, 0])
        post_tile = pl.slice(post, [T, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [T, HC_MULT * HC_MULT], [row0, 0])
        residual_tile = pl.slice(x_flat, [T, HC_MULT, D], [row0, 0, 0])
        active = pl.read(overlay_active_flat, [tile, 1])
        attn_out_tile = pl.create_tensor([T, D], dtype=pl.BF16)
        y_tile = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
        y_tile_initial_flat = pl.reshape(y_tile, [1, T * HC_MULT * D])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_csa_initialize_y_tile'):
            for y_tile_initial_block in pl.range(T * HC_MULT * D // 256):
                y_tile_initial_flat[0:1, y_tile_initial_block * 256:(y_tile_initial_block + 1) * 256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            if T * HC_MULT * D % 256 != 0:
                for y_tile_initial_tail in pl.range(T * HC_MULT * D // 256 * 256, T * HC_MULT * D):
                    pl.write(y_tile_initial_flat, [0, y_tile_initial_tail], pl.cast(0.0, pl.FP32))
        _prefill_sparse_attn_math(q_tile, sparse_tile, bias_tile, mask_tile, attn_sink, cos_tile, sin_tile, wo_a, wo_b, wo_b_scale, attn_out_tile, active)
        hc_post_prefill(attn_out_tile, residual_tile, post_tile, comb_tile, y_tile, active)
        x_out_flat[row0:row0 + T, 0:HC_MULT, 0:D] = y_tile
    return pl.reshape(x_out_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D])

@pl.jit
def prefill_cp_csa_rank(x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], main_state_workspace0: pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], inner_state_workspace0: pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], main_state_workspace1: pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], inner_state_workspace1: pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], compress_state: pl.InOut[pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[MAIN_STATE_MAX_BLOCKS], pl.INT32], inner_compress_state: pl.InOut[pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32], owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32], predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32], query_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32], query_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32], overlay_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], pl.INT32], final_segment_t: pl.Tensor[[1], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[T], pl.INT32], final_win_row_src: pl.Tensor[[T], pl.INT32], final_slot_mapping: pl.Tensor[[T], pl.INT32], leaf_positions_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT32], leaf_main_slots_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_idx_slots_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_main_state_slots_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_inner_state_slots_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_num_tokens_input: pl.Tensor[[LOCAL_PARTS, MAX_COMPRESS_LEAVES], pl.INT32], effective_x_workspace: pl.Tensor[[LOCAL_LEAVES * T, D], pl.BF16], hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16], kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16], tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], main_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, HEAD_DIM], pl.BF16], idx_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, IDX_HEAD_DIM], pl.INT8], scale_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP32], record_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, META_DIM], pl.INT32], main_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], pl.FP32], main_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32], inner_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], pl.FP32], inner_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32], compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32]], my_rank: pl.Scalar[pl.INT32]):
    """Run standalone CP-CSA attention for one rank."""
    return prefill_cp_csa_core(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, hadamard_idx, idx_wq_b, idx_wq_b_scale, idx_weights_proj, inner_wkv, inner_wgate, inner_ape, inner_norm_w, main_state_workspace0, inner_state_workspace0, main_state_workspace1, inner_state_workspace1, compress_state, compress_state_block_table, inner_compress_state, inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, segment_starts_t, segment_lengths_t, segment_active_lengths, owner_segments_t, predecessor_segments, query_positions, query_requests, overlay_positions, overlay_requests, overlay_active_lengths, swa_indices, final_segment_t, reverse_index, owner_rank_table, final_win_seg_src, final_win_row_src, final_slot_mapping, leaf_positions_input, leaf_main_slots_input, leaf_idx_slots_input, leaf_main_state_slots_input, leaf_inner_state_slots_input, leaf_num_tokens_input, effective_x_workspace, hidden_tail_window, kv_tail_window, tail_ready, tail_consumed, main_window, idx_window, scale_window, record_window, main_state_window, main_state_meta_window, inner_state_window, inner_state_meta_window, compact_ready, compact_consumed, attn_sink, wo_a, wo_b, wo_b_scale, x_out, my_rank)

@pl.jit.host
def prefill_cp_csa_test(x_hc: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], compress_state: pl.InOut[pl.Tensor[[CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[CP_SIZE, MAIN_STATE_MAX_BLOCKS], pl.INT32], inner_compress_state: pl.InOut[pl.Tensor[[CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[CP_SIZE, INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[CP_SIZE, ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[CP_SIZE, PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[CP_SIZE, PREFILL_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[CP_SIZE, PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[CP_SIZE, PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[CP_SIZE, IDX_CACHE_MAX_BLOCKS], pl.INT32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32], owner_segments_t: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32], predecessor_segments: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32], query_positions: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32], query_requests: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32], overlay_positions: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], pl.INT32], final_segment_t: pl.Tensor[[1], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[T], pl.INT32], final_win_row_src: pl.Tensor[[T], pl.INT32], final_slot_mapping: pl.Tensor[[T], pl.INT32], leaf_positions_input: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT32], leaf_main_slots_input: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_idx_slots_input: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_main_state_slots_input: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_inner_state_slots_input: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64], leaf_num_tokens_input: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES], pl.INT32], effective_x_workspace: pl.Tensor[[CP_SIZE, LOCAL_LEAVES * T, D], pl.BF16], main_state_workspace0: pl.Tensor[[CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], inner_state_workspace0: pl.Tensor[[CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], main_state_workspace1: pl.Tensor[[CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], inner_state_workspace1: pl.Tensor[[CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], x_out: pl.Out[pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32]]):
    hidden_tail_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    kv_tail_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    tail_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    tail_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    main_buf = pld.alloc_window_buffer([RECORDS_PER_WINDOW, HEAD_DIM], dtype=pl.BF16)
    idx_buf = pld.alloc_window_buffer([RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8)
    scale_buf = pld.alloc_window_buffer([RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP32)
    record_buf = pld.alloc_window_buffer([RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32)
    main_state_buf = pld.alloc_window_buffer([STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], dtype=pl.FP32)
    main_state_meta_buf = pld.alloc_window_buffer([STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32)
    inner_state_buf = pld.alloc_window_buffer([STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], dtype=pl.FP32)
    inner_state_meta_buf = pld.alloc_window_buffer([STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32)
    compact_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    compact_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_tail_window = pld.window(hidden_tail_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
        kv_tail_window = pld.window(kv_tail_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
        tail_ready = pld.window(tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        tail_consumed = pld.window(tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        main_window = pld.window(main_buf, [RECORDS_PER_WINDOW, HEAD_DIM], dtype=pl.BF16)
        idx_window = pld.window(idx_buf, [RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8)
        scale_window = pld.window(scale_buf, [RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP32)
        record_window = pld.window(record_buf, [RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32)
        main_state_window = pld.window(main_state_buf, [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], dtype=pl.FP32)
        main_state_meta_window = pld.window(main_state_meta_buf, [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32)
        inner_state_window = pld.window(inner_state_buf, [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], dtype=pl.FP32)
        inner_state_meta_window = pld.window(inner_state_meta_buf, [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32)
        compact_ready = pld.window(compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        compact_consumed = pld.window(compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        prefill_cp_csa_rank(x_hc[rank], hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, hadamard_idx, idx_wq_b, idx_wq_b_scale, idx_weights_proj, inner_wkv, inner_wgate, inner_ape, inner_norm_w, main_state_workspace0[rank], inner_state_workspace0[rank], main_state_workspace1[rank], inner_state_workspace1[rank], compress_state[rank], compress_state_block_table[rank], inner_compress_state[rank], inner_compress_state_block_table[rank], kv_cache[rank], cmp_kv[rank], cmp_block_table[rank], idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank], segment_starts_t, segment_lengths_t, segment_active_lengths[rank], owner_segments_t[rank], predecessor_segments[rank], query_positions[rank], query_requests[rank], overlay_positions[rank], overlay_requests[rank], overlay_active_lengths[rank], swa_indices[rank], final_segment_t, reverse_index, owner_rank_table, final_win_seg_src, final_win_row_src, final_slot_mapping, leaf_positions_input[rank], leaf_main_slots_input[rank], leaf_idx_slots_input[rank], leaf_main_state_slots_input[rank], leaf_inner_state_slots_input[rank], leaf_num_tokens_input[rank], effective_x_workspace[rank], hidden_tail_window, kv_tail_window, tail_ready, tail_consumed, main_window, idx_window, scale_window, record_window, main_state_window, main_state_meta_window, inner_state_window, inner_state_meta_window, compact_ready, compact_consumed, attn_sink, wo_a, wo_b, wo_b_scale, x_out[rank], rank, device=rank)
