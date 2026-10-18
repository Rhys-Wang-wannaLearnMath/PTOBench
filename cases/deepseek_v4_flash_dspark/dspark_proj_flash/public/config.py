"""Fixed contract for the DeepSeek-V4 DSpark projection FLASH preset."""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
@dataclass(frozen=True)
class DeepSeekV4Config:
    name: str; hidden_size: int; num_attention_heads: int; head_dim: int; qk_rope_head_dim: int; q_lora_rank: int; o_lora_rank: int; o_groups: int; sliding_window: int; rms_norm_eps: float; vocab_size: int
    moe_intermediate_size: int; n_routed_experts: int; n_shared_experts: int; num_experts_per_tok: int; scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]; routed_scaling_factor: float; swiglu_limit: float
    num_hidden_layers: int; num_hash_layers: int; num_nextn_predict_layers: int; compress_ratios: Tuple[int, ...]; index_n_heads: int; index_head_dim: int; index_topk: int; hc_mult: int; hc_sinkhorn_iters: int; hc_eps: float
    max_position_embeddings: int; rope_theta: float; compress_rope_theta: float; rope_factor: float; beta_fast: int; beta_slow: int; original_max_position_embeddings: int; dtype: Literal["bf16", "fp8"]; scale_fmt: Optional[Literal["ue8m0"]]; expert_dtype: Optional[Literal["fp4"]]; scale_dtype: Literal["fp32", "fp8"]; max_batch_size: int
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
FLASH = DeepSeekV4Config(name="flash", hidden_size=4096, num_attention_heads=64, head_dim=512, qk_rope_head_dim=64, q_lora_rank=1024, o_lora_rank=1024, o_groups=8, sliding_window=128, rms_norm_eps=1e-6, vocab_size=129280, moe_intermediate_size=2048, n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=6, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=10.0, num_hidden_layers=43, num_hash_layers=3, num_nextn_predict_layers=1, compress_ratios=(0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0), index_n_heads=64, index_head_dim=128, index_topk=512, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, max_position_embeddings=16384, rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=16.0, beta_fast=32, beta_slow=1, original_max_position_embeddings=65536, dtype="fp8", scale_fmt="ue8m0", expert_dtype="fp4", scale_dtype="fp8", max_batch_size=4)
CASE_CONFIG = FLASH
CASE_ID = "deepseek_v4_flash_dspark.dspark_proj_flash"
DECODE_BATCH = 64; DECODE_SEQ = 2; DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ; PREFILL_BATCH = 1; PREFILL_SEQ = 128; PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ; TP = 4
D = CASE_CONFIG.hidden_size; TARGET_LAYERS = 3; MAIN_HIDDEN_DIM = TARGET_LAYERS * D; FIXTURE_BATCH = DECODE_BATCH // TP; FIXTURE_SEQ = DECODE_SEQ; FIXTURE_TOKENS = FIXTURE_BATCH * FIXTURE_SEQ
