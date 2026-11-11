# Ascend 910B Documentation

15 English technical pages from the fixed CANN 9.0.X (`900`) documentation family,
reviewed against local 910B4 / NpuArch 2201 / CANN 9.0.0. Read the
[baseline](../BASELINE.md) for resource values and unresolved alignment/bank differences.

| Topic | Contents |
| --- | --- |
| [basic_architecture](en/basic_architecture.md) | Compute/storage/movement units and execution model |
| [architecture_220x](en/architecture_220x.md) | 220x data layout, transfers and synchronization constraints |
| [constraints_220x](en/constraints_220x.md) | Local memory sizes, address/alignment and operation constraints |
| [bank_conflicts_220x](en/bank_conflicts_220x.md) | Bank-conflict mechanism and scoped examples; local bank mapping remains unresolved |
| [tiling_headers](en/tiling_headers.md) | Ascend C tiling header interfaces and local include-root compatibility |
| [debugging_overview](en/debugging_overview.md) | Debugging routes and local tool prerequisites |
| [cpu_twin_debugging](en/cpu_twin_debugging.md) | Ascend C CPU twin debug flow; distinct from PyPTO functional simulation |
| [npu_debugging](en/npu_debugging.md) | NPU debug tools and sanitizer meanings |
| [sync_control_overview](en/sync_control_overview.md) | Intra-core ordering and synchronization model |
| [setflag_waitflag](en/setflag_waitflag.md) | Set/Wait ordering and event pairing |
| [pipe_barrier](en/pipe_barrier.md) | Pipeline-local barrier semantics |
| [alloc_event_id](en/alloc_event_id.md) | Dynamic intra-core event allocation and release |
| [fetch_event_id](en/fetch_event_id.md) | Static event lookup and reuse constraints |
| [cross_core_setflag](en/cross_core_setflag.md) | Cross-core signaling modes and counter constraints |
| [cross_core_waitflag](en/cross_core_waitflag.md) | Cross-core wait behavior and local target implementation |

These Ascend C C++ interfaces explain hardware behavior; they are not Python
`pl.*` signatures. Other-product support tables remain scoped source material and
must be filtered before generating 910B4 Agent knowledge. Images, webpage captures,
duplicate Chinese pages and auxiliary audit files are outside this input.
Official English URLs remain in each page; bilingual provenance and prior
corrections are in the [archive](../../Corpus_Archive/20261007_input_cleanup/REPORT.md).
