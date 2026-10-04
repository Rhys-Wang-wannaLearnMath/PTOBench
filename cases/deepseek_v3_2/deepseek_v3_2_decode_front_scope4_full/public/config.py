"""Fixed contract for deepseek_v3_2_decode_front_scope4_full."""

from __future__ import annotations


class DecodeFrontScope4Config:
    """Readable fixed values for the standalone Scope 4 full preset."""

    def __init__(self, *, profile, batch, max_seq_len, num_heads,
                 kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
                 v_head_dim, index_topk, ep_nodes):
        self.profile = profile
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_topk = index_topk
        self.ep_nodes = ep_nodes

    @property
    def qk_head_dim(self):
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def attn_out(self):
        return self.num_heads * self.v_head_dim

    @property
    def cache_rows(self):
        return self.batch * self.max_seq_len

    @property
    def attn_scale(self):
        return 1.0 / (self.qk_head_dim ** 0.5)

    @property
    def builder_args(self):
        return (self.batch, self.max_seq_len, self.num_heads,
                self.kv_lora_rank, self.qk_nope_head_dim,
                self.qk_rope_head_dim, self.v_head_dim, self.index_topk,
                self.ep_nodes)


FULL_PROFILE = DecodeFrontScope4Config(
    profile="full", batch=16, max_seq_len=128, num_heads=128,
    kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
    v_head_dim=128, index_topk=2048, ep_nodes=128,
)
CASE_CONFIG = FULL_PROFILE


# Fixed operator ABI bindings used by Ref, Candidate, Oracle, and evaluator.
BATCH = CASE_CONFIG.batch
MAX_SEQ_LEN = CASE_CONFIG.max_seq_len
NUM_HEADS = CASE_CONFIG.num_heads
KV_LORA_RANK = CASE_CONFIG.kv_lora_rank
QK_NOPE_HEAD_DIM = CASE_CONFIG.qk_nope_head_dim
QK_ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
QK_HEAD_DIM = CASE_CONFIG.qk_head_dim
V_HEAD_DIM = CASE_CONFIG.v_head_dim
ATTN_OUT = CASE_CONFIG.attn_out
INDEX_TOPK = CASE_CONFIG.index_topk
EP_NODES = CASE_CONFIG.ep_nodes
CACHE_ROWS = CASE_CONFIG.cache_rows
ATTN_SCALE = CASE_CONFIG.attn_scale
DTYPE = "bfloat16"
INDEX_DTYPE = "int32"
