> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../../BASELINE.md).

# Run a Current-API Kernel on This Host

The repository's `examples/beginner/hello_world.py` still uses `pl.auto_chunk`,
`TensorSpec(is_output=...)`, and the removed Golden `program/compile_cfg/runtime_cfg`
arguments. It fails before runtime in the current environment. Use this corpus's
[current-API smoke example](../examples/current-smoke.md), copied from the verified audit.

Run from the PyPTO-Lib root:

Save the complete code in [Current-API Smoke Example](../examples/current-smoke.md)
as `current_smoke.py` in the repository root before running the recipe.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PTOAS_ROOT=/data1/home/ziruiwang/ptoas-v0.60
source /usr/local/Ascend/cann-9.0.0/cann/set_env.sh
.venv-official/bin/python current_smoke.py --platform a2a3sim
```

The kernel uses module-level `@pl.jit`, `pl.Out`, a CORE_GROUP scope and tensor slices.
`golden.run` receives `fn=`, `specs=`, `golden_fn=` and one `config=` dict. Output direction
comes from `pl.Out`; `TensorSpec` has no `is_output` argument. It computes `y = x + a`
on a 16×32 FP32 matrix and compares it with the Torch reference.

Expected stages are compile → generate inputs → compute golden → runtime → validate.
When enabled, Benchmark follows validation. Simulator timing is a functional check,
not a hardware latency baseline. Save this example as a file: JIT reads source text,
so a decorated function piped to standard input is not a substitute.

| Platform | Compiler target | This host |
| --- | --- | --- |
| `a2a3sim` | Ascend910B | Minimal simulator tested |
| `a2a3` | Ascend910B | 8×910B4 available as hardware; device execution needs allocated IDs |
| `a5sim` | Ascend950 | Minimal simulator tested; no A5 device certification |
| `a5` | Ascend950 | No matching device on this host |

Platform mapping belongs to PyPTO's RunConfig/compiler implementation; current Golden
has no `golden.runner._backend_for_platform`. The smoke CLI deliberately supports only
simulators. Do not add a device flag to it and assume a device certification happened.

For actual Case execution use PTOBench's selected profile and assigned device set.
Continue with [Golden Harness](../run-and-validate/golden-harness.md),
[Save and Replay](../run-and-validate/save-and-replay.md), or
[L2 Programming](../pypto-coding/l2-programming.md).
