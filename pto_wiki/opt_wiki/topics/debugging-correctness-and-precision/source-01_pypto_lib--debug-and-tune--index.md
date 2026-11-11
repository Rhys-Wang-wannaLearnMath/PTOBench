> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

> **Local recheck — 2026-10-06:** On this host use the pinned interpreter and toolchain described in Installation. Instruction-level simulator collection remains unverified; locating its trace-cleaning module does not certify collection. The generic Golden mean/round-count convention below is separate from PTOBench's clean `median_effective_us` and fixed profile.

# Debug and Tune

Use these guides after a kernel can compile and run. Start with the symptom,
then move from the broadest evidence to the narrowest:

1. Reproduce correctness failures with fixed inputs.
2. Identify whether the problem is compilation, runtime scheduling, numerical
   precision, or performance.
3. Capture task-level evidence before instrumenting an individual kernel.
4. Change one independent variable at a time and revalidate correctness.

## Choose a guide

| Goal | Start here |
|------|------------|
| Diagnose compile errors, runtime failures, hangs, or missing dependencies | [Debugging](source-01_pypto_lib--debug-and-tune--debugging.md) |
| Diagnose numerical drift and choose comparison thresholds | [Precision Tuning](source-01_pypto_lib--debug-and-tune--precision-tuning.md) |
| Measure end-to-end time and inspect the task schedule | [Performance Tuning](../performance-and-optimization/source-01_pypto_lib--debug-and-tune--performance-tuning.md) |
| See how a full decode path was optimized, in order, with the limit found at each step | [DeepSeek V4 Decode Optimization](../model-implementations/source-01_pypto_lib--models--deepseek_v4_flash_mtp--decode_optimization.md) |
| See how a single-card dense model's kernels were tuned, in the order the work happened | [Qwen3-14B Optimization](../model-implementations/source-01_pypto_lib--models--qwen3_14b--optimization.md) |
| Understand how task edges are formed and when the scheduler issues them | [Dependencies and Scheduling](../task-graphs-and-scheduling/source-01_pypto_lib--debug-and-tune--dependency-and-scheduling.md) |
| Fit intermediate tensors in the runtime's ring heaps and measure per-scope peaks | [Ring Heap and Scope Stats](../performance-and-optimization/source-01_pypto_lib--debug-and-tune--ring-heap-and-scope-stats.md) |
| Choose matmul row, N, and K tiles | [Cube Tile Tuning](../performance-and-optimization/source-01_pypto_lib--debug-and-tune--cube-tile-tuning.md) |
| Warm L2 for a weight set the next stage evicts | [L2 Prefetch](../performance-and-optimization/source-01_pypto_lib--debug-and-tune--l2-prefetch.md) |
| Inspect one generated kernel in the operator simulator | [In-Core Simulator Profiling](../performance-and-optimization/source-01_pypto_lib--debug-and-tune--incore-simulator-profiling.md) |
| Partition phases inside a multi-core CCE extern kernel on real hardware | [CCE In-Core Profiling](../performance-and-optimization/source-01_pypto_lib--debug-and-tune--cce-incore-profiling.md) |

## Evidence hierarchy

Prefer evidence that changes the program least:

1. Existing compile reports and validation output.
2. Repeated device benchmarks and chip swimlanes.
3. PMU counters and simulator traces for one kernel.
4. On-device instrumentation added to an extern kernel.

Simulator traces are cycle-accurate for the generated standalone case, but
they do not reproduce the full multi-core schedule or necessarily use real
control inputs. On-device timestamps preserve the real workload, but the
instrumentation changes the kernel ABI and adds overhead. Use the two methods
for different questions rather than treating either as a universal timing
source.

## Keep experiments reproducible

- Save or replay one input set when comparing precision or performance.
- Use the same platform, device topology, runtime configuration, and benchmark
  round count for every candidate.
- Collect repeats as rounds inside one process (`PYPTO_BENCH_ROUNDS`), not as
  repeated invocations of the script, and report the `mean=` field of the
  `[RUN] effective_us` line. Retain the raw samples (`PYPTO_BENCH_RAW=1`).
- Keep generated traces and build products under `build_output/`; do not commit
  them.
- Remove diagnostic instrumentation and rerun validation before submitting a
  production change.
