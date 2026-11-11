# clean_wiki — 本机环境修正版

English documentation input adapted to the installed 910B4 / CANN 9.0.0 /
pinned PyPTO environment. Read [BASELINE](BASELINE.md) for versions, resource
limits, unresolved source ambiguities and the PTOBench action boundary.

| Source | Entry | Retained content |
| --- | --- | --- |
| PyPTO-Lib | [01_pypto_lib](01_pypto_lib/README.md) | 36 source pages and the current-API smoke code as Markdown |
| PyPTO | [02_pypto](02_pypto/README.md) | 186 English pages; 11 installed API bodies |
| Ascend 910B | [03_npu_910b](03_npu_910b/README.md) | 15 English CANN 9.0.X technical pages |

[INPUT_MANIFEST.json](INPUT_MANIFEST.json) explicitly lists workflow inputs and
their hashes. Navigation-only API redirects and README indexes are excluded from
content extraction. Do not ingest every file under pto_wiki.

Audits, helpers, logs, captures, Chinese comparisons and prior change records are
preserved in the [outside-input archive](../Corpus_Archive/20261007_input_cleanup/REPORT.md).
The [original edition](../raw_wiki/README.md) remains immutable.
The [optimized edition](../opt_wiki/README.md) currently
contains only its README. [Knowledge_Workflow](../Knowledge_Workflow/README.md)
defines the future filtering, synthesis and self-contained dynamic loading.

This input preserves technical source material and scoped developer recipes.
The workflow must still remove irrelevant hardware branches, implementation
details and unavailable actions before publishing Agent knowledge.
