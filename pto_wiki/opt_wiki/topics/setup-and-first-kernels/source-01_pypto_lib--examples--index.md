> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](source-BASELINE.md).

> **Applicability note:** The All-reduce row and recipes describe an upstream example absent from this checkout. The other directory examples contain removed Golden arguments and the old CLI flag `--enable-l2-swimlane`; use [the current smoke](source-01_pypto_lib--get-started--first-kernel.md) for a local acceptance check.

# Examples

`examples/` holds small, self-contained kernels arranged as a learning path.
The upstream examples were designed to build synthetic inputs, compute a Torch
golden, run the Golden Harness and exit nonzero on validation failure. The local
directory examples still use removed interfaces and cannot complete that path
in the selected environment. They demonstrate kernel construction and validation; none of
them loads a model checkpoint.

The intended current shape is a `@pl.jit` kernel. The 10 existing directory examples still use old APIs and are not working acceptance commands for this environment. Current examples should split the problem into
`pl.parallel` tiles, do the work inside a `pl.at` core-group scope, describe
inputs and outputs with Golden Harness specs, and compare the result against a
Torch reference. Read them in listed order — each one adds a single idea to the
one above it.

## The catalog

| Level | Example | What it adds |
| --- | --- | --- |
| Beginner | Hello World (`../../../../examples/beginner/hello_world.py`) | The whole path from a tensor signature to a validated result: `pl.parallel` row tiles, `pl.range` column tiles, one `pl.add` |
| Beginner | Matmul (`../../../../examples/beginner/matmul.py`) | M/N tiling with K in a single tile — the first cube example, before a reduction loop |
| Intermediate | GEMM (`../../../../examples/intermediate/gemm.py`) | K tiling: the first tile creates the accumulator with `pl.matmul`, later tiles update it with `pl.matmul_acc` |
| Intermediate | Softmax (`../../../../examples/intermediate/softmax.py`) | A stable reduce-and-broadcast pipeline — `row_max`, `row_sum`, row broadcast |
| Intermediate | LayerNorm (`../../../../examples/intermediate/layer_norm.py`) | Two dependent reductions in one tile, then gamma and beta by broadcast |
| Intermediate | RMSNorm (`../../../../examples/intermediate/rms_norm.py`) | A reduction wider than one tile: chunked accumulation, then a second normalizing pass |
| Intermediate | RoPE (`../../../../examples/intermediate/rope.py`) | Transformer layout — half-head slicing, column broadcast, assembling an output from slices |
| Advanced | GEMM + elementwise (`../../../../examples/advanced/gemm_eltwise.py`) | Fusion: a BF16 matmul with an FP32 accumulator adds its residual without leaving the scope |
| Advanced | Multi-projection (`../../../../examples/advanced/multi_proj.py`) | One `@pl.jit.inline` body reused for Q, K and V without a dispatch per call |
| Advanced | Top-k (`../../../../examples/advanced/topk.py`) | Specialized instructions — `sort32` and chained `mrgsort`, mask gather, tie-aware validation |
| Advanced | All-reduce (`../../examples/advanced/allreduce.py`; unavailable in this snapshot) | L3 distributed execution: window buffers, remote tile loads, notify/wait |

The 10 existing examples declare `a2a3`, `a2a3sim`, `a5`, and `a5sim` in their `--platform`
choices. An accepted `-p a5` is not by itself evidence that the example has been
validated on an A5 device.

## Running an example

Use [the current-API smoke](source-01_pypto_lib--get-started--first-kernel.md) for a functional check:

Save the complete code in [Current-API Smoke Example](source-01_pypto_lib--examples--current-smoke.md)
as `current_smoke.py` in the repository root before running the recipe.

```bash
.venv-official/bin/python current_smoke.py --platform a2a3sim
```

The legacy directory examples expose `--enable-l2-swimlane`, not the newer
`--enable-chip-swimlane` spelling described upstream. Their removed Golden
arguments must also be migrated before those scripts can run. The catalog
remains useful as an algorithm reference.

All-reduce is the only multi-device case. It is written for exactly two ranks
and its CLI requires two device IDs —
[L3 Programming](../distributed-programming/source-01_pypto_lib--pypto-coding--l3-programming.md) is the
reference for every construct it uses:

```text
UPSTREAM RECIPE — target is unavailable in the local checkout.
python examples/advanced/allreduce.py -p a2a3 -d 0,1
python examples/advanced/allreduce.py -p a2a3 -d 0,1 --compile-only
```
