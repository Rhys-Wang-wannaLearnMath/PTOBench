# PyPTO Documentation

186 English pages from the pinned `747e1e4ba99f13dea1a609d893ac7512e85a5897`
documentation, plus 11 installed signature/docstring bodies. All 186 Chinese
counterparts were available for comparison; there were no Chinese-only pages.
The Chinese files and comparison records are preserved outside this input.
Read the [baseline](../BASELINE.md) for local applicability.

| Section | Contents |
| --- | --- |
| [Overview](en/index.md) | Documentation organization |
| [User Guide](en/user/index.md) | Programming model, types, placement, tasks, execution, operators, precision and tutorials |
| [Performance](en/user/performance/index.md) | Task granularity, host overhead, dependencies, memory and profiling meaning |
| [Distributed](en/user/distributed/index.md) | Multi-device semantics and examples; conditional on Case/device permissions |
| [Developer Guide](en/dev/index.md) | IR, passes, code generation and runtime behavior; only source-visible consequences enter Agent knowledge |
| [Reference](en/reference/index.md) | Related PTO ISA material |
| [Installed API](static_api/README.md) | Complete local Python signatures/docstrings replacing unrendered MkDocstrings directives |

API entry pages link to the installed bodies instead of duplicating them.
Compiler-internal IR, illustrative fragments and explicit negative examples are
not additional public Python APIs. Source comparison, local technical fixes and
old verification records are in the
[archive](../../Corpus_Archive/20261007_input_cleanup/REPORT.md).
