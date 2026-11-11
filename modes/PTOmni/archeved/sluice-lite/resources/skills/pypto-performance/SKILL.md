---
name: pypto-performance
description: Apply current PyPTO programming, scheduling, memory, and measurement principles to a PTOBench Candidate.
---

# PyPTO performance work

Use this as domain guidance alongside the Case contract and permissions in `.benchmark_context/`. Start from the actual Candidate, its environment, and public feedback. Do not infer capabilities or tuning settings from another Case, an old CANNBot example, or the host's total hardware.

## Choose a source-backed direction

- Inspect the relevant source and matching SLUICE knowledge card before changing code. Treat both as a possible explanation, not proof that the same mechanism limits this Case.
- Use clean `ptobench evaluate` latency to judge a change. Use a profile only when a specific question about dispatch, dependencies, memory movement, compute, or compilation could change the next step.
- Change one coherent mechanism at a time. Keep a change only when correctness passes and clean latency improves. A profile observation can explain a result, but a faster result alone does not prove the proposed cause.

## Keep the PyPTO levels straight

- Orchestration code schedules work; InCore code describes work executed by a kernel. A `pl.scope()` boundary controls dependencies and resource lifetime; it does not itself fuse kernels.
- Tensor and Tile values have different roles. `pl.slice` describes a Tensor region, `pl.load` transfers Tensor data into a Tile, and `pl.store` writes a Tile back to a Tensor. Do not mix Tensor and Tile operands as if they had identical storage semantics.
- `pl.reshape` is a view, not a request to make transposed data contiguous. Check strides, valid regions, and the consumers before changing shape or layout. Preserve the Case's public input/output ABI and precision contract.

## Reason about loops, tasks, and dependencies

- Use the installed DSL loop forms deliberately: `pl.range` for sequential/carried work, `pl.parallel` only for independent iterations, and `pl.unroll` for small compile-time expansion. Unrolling can enlarge generated code; splitting work can add dispatches.
- Grouping small tasks may reduce dispatch overhead and intermediate writes, but may reduce parallel work and increase live storage. Splitting a task may improve parallel supply while increasing dispatches or repeated loads. Preserve every producer/consumer dependency.
- Larger tiles can improve reuse and reduce repeated transfers, but can increase on-chip storage and worsen tails. Smaller tiles can improve occupancy or tail handling while increasing work and dispatch count. Derive changes from actual shapes and constraints; there is no universal best tile size.

## Preserve tails, memory, and overlap

- `valid_shape` describes the real extent on each axis; it does not create missing input data or guarantee invalid lanes are harmless. Keep output writes in bounds and give reductions the correct identity/padding.
- The current pipeline form is `pl.pipeline(..., stage=...)`. More stages can overlap work, but also replicate buffers and increase memory pressure. Check dependencies, live buffers, and correctness before judging the latency.
- Do not remove explicit dependencies, change memory scope, or bypass a transfer merely to reduce a profile wait. A source-level cleanup or fewer operations is not evidence of a faster kernel.

## Use the installed API and evidence

- Follow the PyPTO API available in the Session environment. Older CANNBot material may name removed APIs or controls; do not copy a setting or syntax until the installed package and Case permissions confirm it exists.
- Read profile facts with their units, scope, and collection mode. Keep runtime, PMU, compiler, and clean-evaluation results distinct. If a fact is unavailable or ambiguous, state that uncertainty and choose a change supported by source or other public evidence.
