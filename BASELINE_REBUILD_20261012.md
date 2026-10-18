# Ref baseline rebuild: `deepseek_v4.synthetic_token_loop_prefill_pro`

The Case previously carried `analyze_auto_scopes_for_deps: true`. The current
Case keeps its `ring_heap` and has no compiler override. Its canonical sealed
asset was absent, so the already reviewed asset tree from
`benchmarks/pypto_lib_bench-v0.2/sealed_assets/deepseek_v4/synthetic_token_loop_prefill_pro`
was restored after matching the current Case's evaluator, Ref, Oracle, Fixture,
config, comparator and metadata. The old `ref_performance.json` was discarded.

The formal command was run on the Case environment with five warmups and ten
measured launches:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python \
  benchmarks/PTOBench/evaluator/prepare_ref.py \
  --case deepseek_v4.synthetic_token_loop_prefill_pro
```

The Ref passed correctness for all 11 declared outputs. The fresh baseline uses
`memory_planner=PYPTO` and `analyze_auto_scopes_for_deps=false` and is recorded
in `sealed_assets/deepseek_v4/synthetic_token_loop_prefill_pro/ref_performance.json`:

- median effective latency: **75,529.468 us**
- mean effective latency: **75,293.6857 us**
- measured samples: **10**
- comparator SHA-256: `42b8c37dee3a6bdce5c595e6e61f56663a5dd2db43c3077ca59b512652da8ab4`

The post-rebuild Session preflight also passed for the canonical Case, Fixture,
Ref cache, comparator and environment binding.
