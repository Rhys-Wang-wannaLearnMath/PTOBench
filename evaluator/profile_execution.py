"""Process-local output-validation policy for managed Agent diagnostics."""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def output_validation(*, enabled: bool):
    if enabled:
        yield
        return

    # Golden has no input-cache-only public option. Its runner already skips
    # validation (including resident readback) when golden_outputs is None.
    # Keep cached inputs, ABI checks, native execution and diagnostics intact;
    # suppress only expected-output loading/computation in this evaluator
    # subprocess. No installed package or per-Case adapter is changed.
    import golden.runner as runner

    original = runner._compute_golden

    def no_expected_outputs(*args, **kwargs):
        print('[PTOBench] Agent profile: output correctness checking disabled.', flush=True)
        return None

    runner._compute_golden = no_expected_outputs
    try:
        yield
    finally:
        runner._compute_golden = original
