from decode_indexer import golden_indexer


def compute(values):
    golden_indexer(values)
    return {name: values[name] for name in ("score", "idx_kv_cache", "idx_kv_scale", "topk_idxs")}
