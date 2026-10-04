"""Maintainer-repaired FLASH fixed model contract; see model_repair.json."""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

@dataclass(frozen=True)
class DeepSeekV4Config:
    """Follows HuggingFace config.json"""
    name: str
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: Literal['softmax', 'sigmoid', 'sqrtsoftplus']
    routed_scaling_factor: float
    swiglu_limit: float
    num_hidden_layers: int
    num_hash_layers: int
    num_nextn_predict_layers: int
    compress_ratios: Tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    max_position_embeddings: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_max_position_embeddings: int
    dtype: Literal['bf16', 'fp8']
    scale_fmt: Optional[Literal['ue8m0']]
    expert_dtype: Optional[Literal['fp4']]
    scale_dtype: Literal['fp32', 'fp8']
    max_batch_size: int

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** (-0.5)

    @property
    def index_nope_head_dim(self) -> int:
        return self.index_head_dim - self.qk_rope_head_dim

    @property
    def index_weights_scale(self) -> float:
        return self.index_head_dim ** (-0.5) * self.index_n_heads ** (-0.5)

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult
FLASH = DeepSeekV4Config(name='flash', hidden_size=4096, num_attention_heads=64, head_dim=512, qk_rope_head_dim=64, q_lora_rank=1024, o_lora_rank=1024, o_groups=8, sliding_window=128, rms_norm_eps=1e-06, vocab_size=129280, moe_intermediate_size=2048, n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=6, scoring_func='sqrtsoftplus', routed_scaling_factor=1.5, swiglu_limit=10.0, num_hidden_layers=43, num_hash_layers=3, num_nextn_predict_layers=1, compress_ratios=(0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0), index_n_heads=64, index_head_dim=128, index_topk=512, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-06, max_position_embeddings=16384, rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=16.0, beta_fast=32, beta_slow=1, original_max_position_embeddings=65536, dtype='fp8', scale_fmt='ue8m0', expert_dtype='fp4', scale_dtype='fp8', max_batch_size=4)
CASE_CONFIG = FLASH
DECODE_BATCH = 8
DECODE_SEQ = 1
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
EP_WORLD_SIZE = 2
EP_RANK = 0
MOE_TOKENS = DECODE_TOKENS
B = DECODE_BATCH
S = DECODE_SEQ
T = MOE_TOKENS
D = CASE_CONFIG.hidden_size
TOPK = CASE_CONFIG.num_experts_per_tok
VOCAB = CASE_CONFIG.vocab_size
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
MOE_INTER = CASE_CONFIG.moe_intermediate_size
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts // 8 * EP_WORLD_SIZE
N_LOCAL = N_EXPERTS_GLOBAL // EP_WORLD_SIZE
RECV_MAX = EP_WORLD_SIZE * MOE_TOKENS
N_ROUTES = T * TOPK
LAYER_ID = 0
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 0.0001
FP32_NEG_INF = -3.4028234663852886e+38
X_HC_SHAPE = (EP_WORLD_SIZE, T, HC_MULT, D)
HC_FFN_FN_SHAPE = (EP_WORLD_SIZE, MIX_HC, HC_DIM)
HC_FFN_SCALE_SHAPE = (EP_WORLD_SIZE, 3)
HC_FFN_BASE_SHAPE = (EP_WORLD_SIZE, MIX_HC)
NORM_W_SHAPE = (EP_WORLD_SIZE, D)
GATE_W_SHAPE = (EP_WORLD_SIZE, N_EXPERTS_GLOBAL, D)
GATE_BIAS_SHAPE = (EP_WORLD_SIZE, N_EXPERTS_GLOBAL)
TID2EID_SHAPE = (EP_WORLD_SIZE, VOCAB, TOPK)
INPUT_IDS_SHAPE = (EP_WORLD_SIZE, T)
ROUTED_W1_SHAPE = (EP_WORLD_SIZE, N_LOCAL, MOE_INTER, D)
ROUTED_W1_SCALE_SHAPE = (EP_WORLD_SIZE, N_LOCAL, MOE_INTER)
ROUTED_W3_SHAPE = ROUTED_W1_SHAPE
ROUTED_W3_SCALE_SHAPE = ROUTED_W1_SCALE_SHAPE
ROUTED_W2_SHAPE = (EP_WORLD_SIZE, N_LOCAL, D, MOE_INTER)
ROUTED_W2_SCALE_SHAPE = (EP_WORLD_SIZE, N_LOCAL, D)
SHARED_W1_SHAPE = (EP_WORLD_SIZE, MOE_INTER, D)
SHARED_W1_SCALE_SHAPE = (EP_WORLD_SIZE, MOE_INTER)
SHARED_W3_SHAPE = SHARED_W1_SHAPE
SHARED_W3_SCALE_SHAPE = SHARED_W1_SCALE_SHAPE
SHARED_W2_SHAPE = (EP_WORLD_SIZE, D, MOE_INTER)
SHARED_W2_SCALE_SHAPE = (EP_WORLD_SIZE, D)
X_NEXT_SHAPE = X_HC_SHAPE
X_HC_DTYPE = 'bfloat16'
HC_FFN_FN_DTYPE = 'float32'
HC_FFN_SCALE_DTYPE = 'float32'
HC_FFN_BASE_DTYPE = 'float32'
NORM_W_DTYPE = 'bfloat16'
GATE_W_DTYPE = 'float32'
GATE_BIAS_DTYPE = 'float32'
TID2EID_DTYPE = 'int32'
INPUT_IDS_DTYPE = 'int64'
ROUTED_W_DTYPE = 'int8'
ROUTED_W_SCALE_DTYPE = 'float32'
SHARED_W_DTYPE = 'int8'
SHARED_W_SCALE_DTYPE = 'float32'
X_NEXT_DTYPE = 'bfloat16'
