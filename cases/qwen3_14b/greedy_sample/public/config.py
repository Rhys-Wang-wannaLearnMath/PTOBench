"""Fixed evaluation contract for qwen3_14b.greedy_sample."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3Config:
    """Source-shaped Qwen3 model configuration used by this Case."""

    name: str
    family: str
    variant: str

    batch: int
    max_seq: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden: int
    intermediate: int
    vocab: int
    real_vocab: int
    num_layers: int
    sampled_ids_pad: int

    eps: float
    q_head_batch: int
    q_head_pad: int
    int8_scale_max: float
    int8_amax_eps: float

    @property
    def kv_hidden(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def hidden_inv(self) -> float:
        return 1.0 / self.hidden

    @property
    def head_dim_inv(self) -> float:
        return 1.0 / self.head_dim

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def half_dim(self) -> int:
        return self.head_dim // 2

    @property
    def q_per_kv(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def q_groups(self) -> int:
        return self.q_per_kv // self.q_head_batch

    @property
    def total_q_groups(self) -> int:
        return self.num_kv_heads * self.q_groups


QWEN3_14B = Qwen3Config(
    name="qwen3-14b",
    family="qwen3",
    variant="14b",
    batch=16,
    max_seq=4096,
    num_heads=40,
    num_kv_heads=8,
    head_dim=128,
    hidden=40 * 128,
    intermediate=17408,
    vocab=152064,
    real_vocab=151936,
    num_layers=40,
    sampled_ids_pad=8,
    eps=1e-6,
    q_head_batch=5,
    q_head_pad=16,
    int8_scale_max=127.0,
    int8_amax_eps=1e-4,
)

CASE_CONFIG = QWEN3_14B


# Fixed greedy-sampling ABI bindings for this Case.
BATCH = CASE_CONFIG.batch
VOCAB = CASE_CONFIG.vocab
REAL_VOCAB = CASE_CONFIG.real_vocab
SAMPLED_IDS_PAD = CASE_CONFIG.sampled_ids_pad
