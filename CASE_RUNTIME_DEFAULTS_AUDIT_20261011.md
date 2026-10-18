# PTOBench compiler-default audit (2026-10-11)

Checked **318** valid `case.json` files under `benchmarks/PTOBench/cases` and all environment JSON runtime blocks.

| Parameter | Effective baseline | Evidence |
| --- | --- | --- |
| `memory_planner` | **PYPTO** | `Evaluator` maps the public default to `MemoryPlanner.PYPTO`; no Case or environment sets another planner. |
| `analyze_auto_scopes_for_deps` | **false** | `RunConfig` receives false by default; no Case or environment override remains after the fix below. |

The only violating Case found before the fix was `deepseek_v4.synthetic_token_loop_prefill_pro`, which explicitly set `analyze_auto_scopes_for_deps: true`. That key was removed; its `ring_heap` remains unchanged. Full post-fix scan reports zero overrides and all valid Cases conform.

- Valid Case JSON: **318**
- Malformed Case JSON: **0** (these are historical empty/misplaced directories, not runnable Cases)
- Remaining planner/scope overrides: **0**
