# Comparator audit

This is the manual backfill audit required by
the migration workflow's `COMPARATOR_POLICY.md` when nested in pypto-lib. It covers every Case present on
2026-09-19. The upstream comparison target is
`https://github.com/hw-native-sys/pypto-lib.git` at
`312fe5abb4874135be22e954fcd9273e5c9dfa2d` (retrieved 2026-09-19).

The migrated source, ABI, outputs, shapes/layouts, dtypes, state, Fixture input
distribution, and CPU Oracle are fixed. `ADOPT` below therefore means only
that the upstream comparator change is applicable to that fixed contract and
is representable by the existing outputs. `KEEP_SELECTED` means at least one
of those assumptions does not hold. Sibling preset rows were checked against
their own output contracts even when they share one source.

## Evidence groups

- `V32`: the last exact upstream revisions of the removed DeepSeek v3.2
  workloads retain the selected rules: decode back and prefill back use
  `3e-3/3e-3`, decode front uses `4e-3/4e-3`, scope 1 uses `2e-2/2e-2`, and
  scope 2 uses `2e-3/1.0`. There is no independent later comparator change to
  adopt for the fixed outputs.
- `V4-COMBINE`: the last exact revisions before removal retain
  `1e-3/1e-3`. Current successors integrate or rename workload outputs, so
  their additional comparison behavior is not applicable to the fixed Cases.
- `V4-COMPRESSOR`: current successors change output/state contracts and add
  special-value behavior tied to those contracts. Those changes require new
  ABI, state, Fixture, or golden semantics; the selected per-output ratio
  rules are retained.
- `V4-HC-POST`: current HC-post implementations in the DeepSeek V4 successor
  families retain `1e-3/1e-3` for the corresponding numeric output.
- `V4-HC-PRE`: later HC-pre ratio rules accompany FP32 input, flattened shapes,
  changed input distribution, and changed golden accumulation. They are one
  coupled group and are not applicable to the fixed BF16 Case.
- `V4-LM`: current LM-head successors retain `1e-3/1e-3` for `logits` but add
  or change other outputs. Only the unchanged `logits` rule applies to these
  fixed single-output Cases.
- `V4-MOE`: current routed-expert/combine successors integrate `x_next` and
  use comparisons tied to changed output meaning. Those groups require source,
  ABI, Fixture, and golden changes and are not applicable here.
- `V4-OPROJ`: the current DSpark successor's exact comparison belongs to a
  quantized/distributed projection with different inputs and output semantics;
  it is not applicable to the selected BF16 grouped projection.
- `QWEN-GREEDY`: current `models/qwen3_14b/greedy_sample.py` still invokes the
  official runner with `rtol=0` and `atol=0`; the selected strict numeric rule
  is unchanged.
- `QWEN-DECODE`: current `models/qwen3_14b/decode_fwd.py:2012` explains that
  BF16 one-ULP differences can exceed `3e-3` and uses
  `ratio_allclose(rtol=3e-3, atol=3e-3, max_error_ratio=0.02)` at line 2031.
  The fixed Case has the same `out` meaning, dtype, and comparison population,
  so the complete method/tolerance/ratio group is adopted.

### Exact evidence index

Unless a historical revision is written explicitly below, the latest evidence
is read at the fixed `upstream/main` commit named at the top of this document.
The selected side is the byte-exact `sealed_assets/<family>/<case>/official/`
source plus that Case's pre-reconciliation `public/comparator.json`; sibling
rows reuse a source only after checking their own output ABI.

| Evidence | Selected/last-exact evidence | Latest/successor evidence | Semantic finding |
| --- | --- | --- | --- |
| `V32` | commit `7fd3bafcf1d80f52ef6d75ce36c67aec40c215c3`: `models/deepseek_v3_2/decode_back.py:302-303`, `decode_front.py:882-883`, `prefill_back.py:340-341`; scope rules are also fixed by their preserved v0.1 comparator contracts | removal commit `63b05612010e4a4881ef924ec0925a08c26c1d71`; no identifiable current one-to-one successors for these fixed outputs | Last exact runners retain `3e-3/3e-3`, `4e-3/4e-3`, and `3e-3/3e-3`; no later independent group applies. Scope 1/2 keep their selected `2e-2/2e-2` and `2e-3/1.0` rules. |
| `V4-COMBINE` | commit `6b8a92de7ea3a42208186df0a89da0d81247e512`, `models/deepseek/v4/combine.py:266-267`; dispatch-combine last exact predecessor `2d6c0f6399e6c68191387c00a26f66cffe6ed533`, `models/deepseek/v4/moe_dispatch_combine.py:329-330` | current `models/deepseek_v4_flash_dspark/moe.py:1412-1413` and `models/deepseek_v4_pro/moe.py:1293-1294` are integrated MoE successors | Numeric `1e-3/1e-3` remains, while integrated output meaning is not a portable independent change. |
| `V4-COMPRESSOR` | each preserved `official/original.py` and selected four-output comparator JSON | `models/deepseek_v4_flash_dspark/decode_compressor_ratio4.py:658-703`, `models/deepseek_v4_flash_mtp/decode_compressor_ratio4.py:546-579`, and `models/deepseek_v4_pro/decode_compressor_ratio4.py:529-561` | Latest rules rename/change cache/state outputs and add `ignore_nan` under changed contracts. Those fields form coupled groups requiring ABI/state/Fixture/golden changes, so they are not partially adopted. |
| `V4-HC-POST` | preserved decode/prefill HC-post official source and selected `y` comparator | `models/deepseek_v4_flash_dspark/hc_post.py:232-233`, `models/deepseek_v4_flash_mtp/hc_post.py:230-231`, and `models/deepseek_v4_pro/hc_post.py:249-250`; V4.1's different `output` contract is at `models/deepseek_v4_1_flash/hc_post.py:139-155` | The matching fixed numeric HC-post successors retain `1e-3/1e-3`; the V4.1 ratio rule belongs to a different output contract. |
| `V4-HC-PRE` | preserved BF16 three-output HC-pre source and selected `4e-3/4e-3` rules | `models/deepseek_v4_flash_dspark/hc_pre.py:528-570`, `models/deepseek_v4_flash_mtp/hc_pre.py:387-427`, and `models/deepseek_v4_pro/hc_pre.py:843-902` | Latest ratio rules depend on FP32 inputs, flattened shapes, changed distributions and golden accumulation; the whole coupled group is inapplicable. |
| `V4-LM` | preserved two-rank single-`logits` source and selected comparator | `models/deepseek_v4_flash_dspark/lm_head.py:636-637`, `models/deepseek_v4_flash_mtp/lm_head.py:598,693-694`, and `models/deepseek_v4_pro/lm_head.py:585,673-674` | The matching `logits` tolerance stays `1e-3/1e-3`; other latest outputs/aggregation do not exist in the fixed Case. |
| `V4-MOE` | preserved routed-expert/combine and dispatch-combine sources plus their selected output rules | integrated successors `models/deepseek_v4_flash_dspark/moe.py:1412-1413` and `models/deepseek_v4_pro/moe.py:1293-1294` | Latest integrated `x_next` behavior changes source, output meaning, ABI, Fixture and golden together. No isolated comparator-only group applies. |
| `V4-OPROJ` | preserved BF16 grouped projection source and selected `attn_out: 1e-3/1e-3` | `models/deepseek_v4_flash_dspark/decode_o_proj.py:797-798` uses exact runner tolerances for a quantized/distributed contract; current PRO uses `output_projection.py` with different inputs/layout | The exact rule cannot be separated from the successor's changed quantization, topology and output semantics. |
| `QWEN-GREEDY` | preserved source and selected zero-tolerance numeric comparator | `models/qwen3_14b/greedy_sample.py:169-170` | Latest remains `rtol=0`, `atol=0`; unchanged and applicable. |
| `QWEN-DECODE` | preserved runner's strict `3e-3/3e-3` rule | `models/qwen3_14b/decode_fwd.py:2012-2031` | Adopt the atomic method-plus-ratio change: same `rtol/atol`, `max_error_ratio=0.02`, same BF16 `out` assumptions, no workload change. |

## Case decisions

| Case | Evidence | Decision | Final comparator |
| --- | --- | --- | --- |
| `deepseek_v3_2.deepseek_v3_2_decode_back` | V32 | KEEP_SELECTED (unchanged) | `out: allclose(3e-3, 3e-3)` |
| `deepseek_v3_2.deepseek_v3_2_decode_front` | V32 | KEEP_SELECTED (unchanged) | all 4 outputs: `allclose(4e-3, 4e-3)` |
| `deepseek_v3_2.deepseek_v3_2_decode_front_scope1` | V32 | KEEP_SELECTED (unchanged) | all 6 outputs: `allclose(2e-2, 2e-2)` |
| `deepseek_v3_2.deepseek_v3_2_decode_front_scope2` | V32 | KEEP_SELECTED (unchanged) | all 5 outputs: `allclose(2e-3, 1.0)` |
| `deepseek_v3_2.deepseek_v3_2_prefill_back` | V32 | KEEP_SELECTED (unchanged) | `out: allclose(3e-3, 3e-3)` |
| `deepseek_v4.combine_demo` | V4-COMBINE | KEEP_SELECTED (unchanged) | `ffn_out: allclose(1e-3, 1e-3)` |
| `deepseek_v4.combine_flash` | V4-COMBINE | KEEP_SELECTED (unchanged) | `ffn_out: allclose(1e-3, 1e-3)` |
| `deepseek_v4.combine_pro` | V4-COMBINE | KEEP_SELECTED (unchanged) | `ffn_out: allclose(1e-3, 1e-3)` |
| `deepseek_v4.compressor_ratio4_demo` | V4-COMPRESSOR | KEEP_SELECTED | `kv: ratio(0.0078125,1e-4,0); kv_cache: ratio(0.0078125,1e-4,1.220703125e-6); states: ratio(1e-3,1e-3,0)` |
| `deepseek_v4.compressor_ratio4_flash` | V4-COMPRESSOR | KEEP_SELECTED | same four per-output rules as demo |
| `deepseek_v4.compressor_ratio4_pro` | V4-COMPRESSOR | KEEP_SELECTED | same four per-output rules as demo |
| `deepseek_v4.decode_hc_post_b16` | V4-HC-POST | KEEP_SELECTED (unchanged) | `y: allclose(1e-3, 1e-3)` |
| `deepseek_v4.decode_hc_pre_b16` | V4-HC-PRE | KEEP_SELECTED | all 3 outputs: `allclose(4e-3, 4e-3)` |
| `deepseek_v4.decode_moe_expert` | V4-MOE | KEEP_SELECTED | both outputs: `allclose(1e-3, 1e-3)` |
| `deepseek_v4.decode_o_proj_grouped` | V4-OPROJ | KEEP_SELECTED | `attn_out: allclose(1e-3, 1e-3)` |
| `deepseek_v4.hc_post_demo` | V4-HC-POST | KEEP_SELECTED (unchanged) | `y: allclose(1e-3, 1e-3)` |
| `deepseek_v4.hc_post_flash` | V4-HC-POST | KEEP_SELECTED (unchanged) | `y: allclose(1e-3, 1e-3)` |
| `deepseek_v4.hc_post_pro` | V4-HC-POST | KEEP_SELECTED (unchanged) | `y: allclose(1e-3, 1e-3)` |
| `deepseek_v4.lm_head_tp2_demo` | V4-LM | KEEP_SELECTED (unchanged) | `logits: allclose(1e-3, 1e-3)` |
| `deepseek_v4.lm_head_tp2_flash` | V4-LM | KEEP_SELECTED (unchanged) | `logits: allclose(1e-3, 1e-3)` |
| `deepseek_v4.lm_head_tp2_pro` | V4-LM | KEEP_SELECTED (unchanged) | `logits: allclose(1e-3, 1e-3)` |
| `deepseek_v4.moe_combine_demo` | V4-MOE | KEEP_SELECTED | `ffn_out: allclose(1e-3, 1e-3)` |
| `deepseek_v4.moe_combine_flash` | V4-MOE | KEEP_SELECTED | `ffn_out: allclose(1e-3, 1e-3)` |
| `deepseek_v4.moe_combine_pro` | V4-MOE | KEEP_SELECTED | `ffn_out: allclose(1e-3, 1e-3)` |
| `deepseek_v4.moe_dispatch_combine_demo` | V4-MOE | KEEP_SELECTED | both outputs: `allclose(1e-3, 1e-3)` |
| `deepseek_v4.moe_dispatch_combine_flash` | V4-MOE | KEEP_SELECTED | both outputs: `allclose(1e-3, 1e-3)` |
| `deepseek_v4.moe_dispatch_combine_pro` | V4-MOE | KEEP_SELECTED | both outputs: `allclose(1e-3, 1e-3)` |
| `deepseek_v4.prefill_hc_post_demo` | V4-HC-POST | KEEP_SELECTED (unchanged) | `y: allclose(1e-3, 1e-3)` |
| `deepseek_v4.prefill_hc_post_flash` | V4-HC-POST | KEEP_SELECTED (unchanged) | `y: allclose(1e-3, 1e-3)` |
| `deepseek_v4.prefill_hc_post_pro` | V4-HC-POST | KEEP_SELECTED (unchanged) | `y: allclose(1e-3, 1e-3)` |
| `qwen3_14b.greedy_sample` | QWEN-GREEDY | KEEP_SELECTED (unchanged) | `sampled_ids: allclose(0, 0)` |
| `qwen3_14b.qwen3_14b_decode` | QWEN-DECODE | ADOPT | `out: ratio_allclose(3e-3, 3e-3, 0.02)` |

All 32 final rules are fully represented by `public/comparator.json`; no Case
requires custom Python comparison semantics. Each sealed Case evaluator now
independently reads and strictly validates that JSON and executes its declared
standard method; common evaluator code does not interpret comparator fields.
Comparator-only adoption for Qwen decode does not authorize changes to its
source, Fixture inputs, golden, dtype, or environment. It does require fresh
Ref certification and invalidates prior Candidate results and acceptance
workspaces.

## Backfill verification

The 2026-09-19 backfill completed with the repository's declared benchmark
Python and Case environment profiles; no environment profile or installed
tool version was changed.

- All 32 comparator JSON files passed their owning Case evaluator's strict
  local parser and method implementation. All 32 evaluators reject unknown or
  extra fields and output drift; none calls a common comparator factory.
- All 32 Refs passed fresh CPU-Oracle certification. Every certification is
  `ready`, records the SHA-256 of its current comparator JSON, and has positive
  Ref timing samples.
- All 32 canonical Candidate Seeds passed correctness from newly initialized
  workspaces.
- `qwen3_14b.qwen3_14b_decode`, formerly pending and missing its Candidate
  Seed, was mechanically published from its certified Ref. The mapped
  `ref_npu.py` and `candidate.py` hashes are both
  `68d06fcad3e50c307280bb9af5c1377c1d44afdb6ce12f3197a8813c30e8dde5`.
- That Qwen Case passed initial clean, runtime profile, hardware profile,
  compile profile, and final clean. Both clean runs are official phase-all
  results with positive timing samples; all profile runs are diagnostic-only
  correctness results with nonempty isolated artifacts.

## Case-local comparator execution maintenance

On 2026-09-19 comparator execution ownership was moved out of
`evaluator/common.py` and restored to every sealed Case `evaluate.py`. The
public JSON rules, source, Ref/Candidate implementation, Fixture inputs,
golden/Oracle, ABI, dtype, shape/layout, state, environment, and toolchain were
not changed.

- All 32 Case evaluators independently read `public/comparator.json`, require
  exact top-level and correctness-output coverage, validate exact method fields
  and finite numeric ranges, and locally execute `exact`, `allclose`, and
  `ratio_allclose`. No evaluator calls a shared comparator factory.
- The local `ratio_allclose` decision was checked against the official
  `golden.ratio_allclose` over boundary populations, error counts, and allowed
  ratios, including the Qwen 2% rule; all PASS/FAIL decisions matched.
- All 32 Refs passed fresh plain `bench prepare-ref` runs without Fixture-input
  regeneration. Certification, cached Ref outputs, and positive performance
  samples were refreshed under the Case-local implementations.
- All 32 canonical Candidate Seeds then passed from fresh isolated workspaces:
  initial official phase-all clean, `runtime`, `hardware`, and `compile`
  diagnostic correctness, and final official phase-all clean. Every mode had
  a PASS result and nonempty isolated artifacts; both clean runs had positive
  timing samples.
- Evaluator tests passed (`12 passed`), all 32 local comparator implementations
  passed AST/ownership checks, all comparator/certification hashes matched, and
  all seven migration Skills passed `skill-creator` validation.
