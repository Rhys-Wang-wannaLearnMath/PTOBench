# Current-API Smoke Example

This example uses the installed JIT and Golden APIs. Save the complete Python block as
`current_smoke.py` in the PyPTO-Lib repository root before using the developer recipe:

Save the complete code in [Current-API Smoke Example](../examples/current-smoke.md)
as `current_smoke.py` in the repository root before running the recipe.

```bash
.venv-official/bin/python current_smoke.py --platform a2a3sim
```

The existing audit checked this minimal simulator path; it does not certify device latency
or arbitrary tutorials. PTOBench Agent execution uses the public `ptobench` CLI and its
fixed Case contract. The Golden wrapper here is a developer example.

```python
"""Minimal kernel using the documented current JIT and Golden APIs."""
import argparse
from pathlib import Path
import torch
import pypto.language as pl
from golden import run, TensorSpec, ScalarSpec

@pl.jit
def documented_add(
    x: pl.Tensor[[16, 32], pl.FP32],
    a: pl.Scalar[pl.FP32],
    y: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
):
    for r in pl.parallel(0, 16, 8):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='documented_add'):
            y[r:r + 8, 0:32] = pl.add(x[r:r + 8, 0:32], a)
    return y

def golden_add(values):
    values['y'][:] = values['x'] + values['a']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', choices=['a2a3sim', 'a5sim'], default='a2a3sim')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--save-data', action='store_true')
    parser.add_argument('--golden-data')
    parser.add_argument('--runtime-dir')
    args = parser.parse_args()
    result = run(fn=documented_add,
                 specs=[TensorSpec('x', [16, 32], torch.float32, init_value=torch.randn),
                        ScalarSpec('a', torch.float32, 1.0),
                        TensorSpec('y', [16, 32], torch.float32)],
                 golden_fn=golden_add,
                 config={'platform': args.platform},
                 compile_only=args.compile_only,
                 save_data=args.save_data,
                 golden_data=args.golden_data,
                 runtime_dir=args.runtime_dir)
    print('AUDIT_RESULT', result)
    print('AUDIT_WORK_DIR', result.work_dir)
    raise SystemExit(0 if result.passed else 1)
```
