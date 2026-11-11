# PyPTO-Lib Documentation

The frozen official PyPTO-Lib edition, corrected only where local compatibility
requires it. All 36 source pages are English; the relevant current-API example
is preserved as [Markdown](examples/current-smoke.md). Read the [baseline](../BASELINE.md).

| Section | Contents |
| --- | --- |
| [Overview](index.md) | Library organization and reading paths |
| [Get Started](get-started/index.md) | Environment and a current-API minimal kernel |
| [Coding](pypto-coding/index.md) | PyPTO source style, loops, L2/L3 execution and extern conventions |
| [Examples](examples/index.md) | Kernel algorithms; legacy example compatibility is explicitly marked |
| [Run and Validate](run-and-validate/index.md) | Compiler/runtime flow, Golden and data replay |
| [Debug and Tune](debug-and-tune/index.md) | Precision, dependencies, task granularity, tiling, memory and profiling |
| [Models](models/index.md) | Model architecture and historical optimization examples |

Historical timings, other-device thresholds and developer runner recipes retain
their stated scope. They are not PTOBench Agent permissions or local performance
certification. Model solutions require case-level leakage filtering before use.
Prior corrections and helper scripts are in the
[archive](../../Corpus_Archive/20261007_input_cleanup/REPORT.md).
