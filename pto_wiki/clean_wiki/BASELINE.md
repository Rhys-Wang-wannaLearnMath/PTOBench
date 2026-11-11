# Local Environment and Applicability

Baseline recorded on 2026-10-05 and read-only rechecked on 2026-10-07. Package
version `0.1.0` alone does not identify the installed compiler or runtime.

| Component | Local baseline |
| --- | --- |
| Hardware | 8 x Ascend910B4, 32 GiB HBM per device; aarch64 |
| Architecture | A2 / NpuArch 2201 / AIC-C-220; separate AIC and AIV |
| Compute cores per device | 20 AIC / 40 AIV; generic compiler 24/48 is not this SKU |
| L2 per device | 100663296 bytes = 96 MiB |
| UB per AIV | Physical/configured 192 KiB; pinned compiler Vec budget 184 KiB |
| Other local memories | L1 512 KiB; L0A and L0B 64 KiB each; L0C 128 KiB; BiasTable 1 KiB |
| Python | `.venv-official/bin/python`, 3.10.20 |
| PyPTO | 0.1.0; `747e1e4ba99f13dea1a609d893ac7512e85a5897` |
| simpler | 0.1.0; `4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0` |
| PTOAS | 0.60; `/data1/home/ziruiwang/ptoas-v0.60` |
| Managed PTO ISA | `a8040450238f162985d8b596fbebeb54bfba2bf5` |
| CANN | 9.0.0; `/usr/local/Ascend/cann-9.0.0/cann-9.0.0` |
| CANN environment script | `/usr/local/Ascend/cann-9.0.0/cann/set_env.sh` |
| Driver | 25.5.0 |
| Torch / NumPy | 2.6.0+cpu / 2.2.6; `torch.float8_e8m0fnu` is unavailable |
| Developer tools | msprof, mssanitizer and msdebug installed; gdb not found on the checked PATH |

Physical capacity, compiler budget and actual remaining allocation space are
different. Padding, liveness and kernel reservations can reduce usable space.
`a2a3` is a backend family, not an L2 size; internal `a3` names alone do not identify
a physical 910C. A5 simulator examples do not establish A5 hardware availability.

## Reading scope

This is the English source input for a later offline knowledge workflow, rather
than the final Agent context. Retained cross-platform tables and developer recipes
have their original scope; they must pass the workflow's environment and action
filter before becoming Agent knowledge. Historical tuning timings and gains are
not remeasured local thresholds. Standalone repository examples may still use old
interfaces; the relevant pages mark them. Installed API bodies are in
[02_pypto/static_api](02_pypto/static_api/README.md).

Ascend C C++ APIs, PTO IR and public Python `pl.*` calls are different interfaces.
Never translate an Ascend C API signature into an assumed Python API. PyPTO
`pl.system` helpers must be checked in that namespace. CANN `900` source pages are
the 9.0.X documentation family, not patch-by-patch installation certification.

Two source ambiguities remain: Fixpipe buffer alignment is described as 64 and
128 bytes in different chapters; the document's 48-bank UB example differs from
the platform INI's 64-bank fields. Neither establishes a universal alignment rule
or the local visible bank mapping. Do not derive 256 KiB usable UB from those fields.

## PTOBench use

Agent actions are Candidate/helper changes inside a registered workspace and the
public `ptobench evaluate`, `profile runtime`, `profile pmu`, `profile compiler`,
and `fork` operations, subject to the current Session. Public benchmark context,
fixed config/comparator and evaluation controls are read-only. Sealed assets are
excluded. Tool installation does not grant permission to bypass the Bench CLI.

Clean evaluate is the formal correctness/performance result; profiling results
are diagnostic. The Bench default is 5 warmup / 10 measured rounds, distinct from
generic Golden's 5 / 100 and mean-oriented examples. Fixed Case configuration is
authoritative. evaluate/profile reserve an exploration slot even on failure or
interruption; fork does not. Device count and profiling availability come from
the public Case/Session context; local multi-card hardware does not make every
Case multi-device. CCE extern edits require an explicit applicable permission.

Earlier minimal simulator and compile checks cover only their recorded examples,
not every tutorial, multi-card execution or current device performance. This
cleanup did not dispatch an NPU workload or change installed dependencies.

Build provenance and previous audits are archived outside the input in
[Corpus_Archive](../Corpus_Archive/20261007_input_cleanup/REPORT.md).
