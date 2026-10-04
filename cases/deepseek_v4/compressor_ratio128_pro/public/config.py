"""Fixed DeepSeek-V4 compressor-ratio128 PRO contract."""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

@dataclass(frozen=True)
class DeepSeekV4Config:
    name: str; hidden_size: int; num_attention_heads: int; head_dim: int; qk_rope_head_dim: int
    q_lora_rank: int; o_lora_rank: int; o_groups: int; sliding_window: int; rms_norm_eps: float; vocab_size: int
    moe_intermediate_size: int; n_routed_experts: int; n_shared_experts: int; num_experts_per_tok: int
    scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]; routed_scaling_factor: float; swiglu_limit: float
    num_hidden_layers: int; num_hash_layers: int; num_nextn_predict_layers: int; compress_ratios: Tuple[int, ...]
    index_n_heads: int; index_head_dim: int; index_topk: int; hc_mult: int; hc_sinkhorn_iters: int; hc_eps: float
    max_position_embeddings: int; rope_theta: float; compress_rope_theta: float; rope_factor: float
    beta_fast: int; beta_slow: int; original_max_position_embeddings: int
    dtype: Literal["bf16", "fp8"]; scale_fmt: Optional[Literal["ue8m0"]]; expert_dtype: Optional[Literal["fp4"]]
    scale_dtype: Literal["fp32", "fp8"]; max_batch_size: int
    @property
    def nope_head_dim(self): return self.head_dim - self.qk_rope_head_dim
    @property
    def softmax_scale(self): return self.head_dim ** -0.5
    @property
    def index_nope_head_dim(self): return self.index_head_dim - self.qk_rope_head_dim
    @property
    def index_weights_scale(self): return self.index_head_dim ** -0.5 * self.index_n_heads ** -0.5
    @property
    def hc_dim(self): return self.hc_mult * self.hidden_size
    @property
    def mix_hc(self): return (2 + self.hc_mult) * self.hc_mult

PRO = DeepSeekV4Config("pro", 7168, 128, 512, 64, 1536, 1024, 16, 128, 1e-6, 129280,
    3072, 384, 1, 6, "sqrtsoftplus", 2.5, 10.0, 61, 3, 1,
    (128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0),
    64, 128, 1024, 4, 20, 1e-6, 1048576, 10000.0, 160000.0, 16.0, 32, 1, 65536,
    "fp8", "ue8m0", None, "fp8", 4)
CASE_CONFIG = PRO
FLASH = CASE_CONFIG

DECODE_BATCH = 8; DECODE_SEQ = 1
B = DECODE_BATCH; S = DECODE_SEQ
EPS = CASE_CONFIG.rms_norm_eps; D = CASE_CONFIG.hidden_size; HEAD_DIM = CASE_CONFIG.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM; ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
NOPE_HEAD_DIM = CASE_CONFIG.nope_head_dim; MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
COMPRESS_RATIO = 128; COFF = 1; OUT_DIM = COFF * HEAD_DIM; STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO; START_POS = COMPRESS_RATIO - 1
