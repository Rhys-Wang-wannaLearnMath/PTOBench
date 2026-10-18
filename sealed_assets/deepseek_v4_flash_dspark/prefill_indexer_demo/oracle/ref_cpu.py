"""Independent CPU oracle with import-time captured local dependencies."""
from prefill_indexer import golden_prefill_indexer


def compute(values):
    golden_prefill_indexer(values)
    return {name: values[name].clone() for name in (
        "score", "idx_kv_cache", "idx_kv_scale", "topk_idxs",
    )}
