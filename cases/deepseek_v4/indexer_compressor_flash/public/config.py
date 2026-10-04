"""Fixed DeepSeek-V4 indexer compressor FLASH contract."""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

@dataclass(frozen=True)
class DeepSeekV4Config:
    name: str; hidden_size: int; num_attention_heads: int; head_dim: int; qk_rope_head_dim: int; q_lora_rank: int; o_lora_rank: int; o_groups: int; sliding_window: int; rms_norm_eps: float; vocab_size: int
    moe_intermediate_size: int; n_routed_experts: int; n_shared_experts: int; num_experts_per_tok: int; scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]; routed_scaling_factor: float; swiglu_limit: float
    num_hidden_layers: int; num_hash_layers: int; num_nextn_predict_layers: int; compress_ratios: Tuple[int, ...]
    index_n_heads: int; index_head_dim: int; index_topk: int; hc_mult: int; hc_sinkhorn_iters: int; hc_eps: float
    max_position_embeddings: int; rope_theta: float; compress_rope_theta: float; rope_factor: float; beta_fast: int; beta_slow: int; original_max_position_embeddings: int
    dtype: Literal["bf16", "fp8"]; scale_fmt: Optional[Literal["ue8m0"]]; expert_dtype: Optional[Literal["fp4"]]; scale_dtype: Literal["fp32", "fp8"]; max_batch_size: int
    @property
    def nope_head_dim(self) -> int: return self.head_dim - self.qk_rope_head_dim
    @property
    def softmax_scale(self) -> float: return self.head_dim ** -0.5
    @property
    def index_nope_head_dim(self) -> int: return self.index_head_dim - self.qk_rope_head_dim
    @property
    def index_weights_scale(self) -> float: return self.index_head_dim ** -0.5 * self.index_n_heads ** -0.5
    @property
    def hc_dim(self) -> int: return self.hc_mult * self.hidden_size
    @property
    def mix_hc(self) -> int: return (2 + self.hc_mult) * self.hc_mult

FLASH = DeepSeekV4Config("flash", 4096, 64, 512, 64, 1024, 1024, 8, 128, 1e-6, 129280, 2048, 256, 1, 6, "sqrtsoftplus", 1.5, 10.0, 43, 3, 1, (0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0), 64, 128, 512, 4, 20, 1e-6, 16384, 10000.0, 160000.0, 16.0, 32, 1, 65536, "fp8", "ue8m0", "fp4", "fp8", 4)
CASE_CONFIG = FLASH

DECODE_BATCH = 8
DECODE_SEQ = 1
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
B = DECODE_BATCH
S = DECODE_SEQ
D = CASE_CONFIG.hidden_size
HEAD_DIM = CASE_CONFIG.index_head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
NOPE_HEAD_DIM = CASE_CONFIG.index_nope_head_dim
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
EPS = CASE_CONFIG.rms_norm_eps
HEAD_DIM_INV = 1.0 / HEAD_DIM
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
START_POS = COMPRESS_RATIO - 1
ROTATE = True
