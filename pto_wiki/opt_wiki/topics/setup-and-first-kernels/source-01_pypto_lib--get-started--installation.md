> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](source-BASELINE.md).

> **Applicability note:** Local paths and installed versions below were rechecked on 2026-10-06; fresh-install recipes are preserved in the unaltered source.

# Installation and Environment

## Existing host: verified paths and versions

Use the existing application environment from the PyPTO-Lib repository root:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PTOAS_ROOT=/data1/home/ziruiwang/ptoas-v0.60
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0/cann
source "$CANN_ROOT/set_env.sh"
.venv-official/bin/python -c 'import pypto, torch; print(pypto.__file__, torch.__version__)'
```

| Component | This installation |
| --- | --- |
| Host | aarch64, Huawei Cloud EulerOS 2.0 |
| Device | 8 × Ascend910B4, 32 GiB HBM each; 20 AIC / 40 AIV, 96 MiB L2 per device |
| Python | `.venv-official/bin/python`, 3.10.20 |
| PyPTO | 0.1.0 from commit `747e1e4ba99f13dea1a609d893ac7512e85a5897` |
| simpler | 0.1.0 from the pinned runtime `4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0` |
| PTOAS | 0.60, `/data1/home/ziruiwang/ptoas-v0.60/bin/ptoas` |
| Managed PTO ISA | `a8040450238f162985d8b596fbebeb54bfba2bf5` |
| Torch | 2.6.0+cpu; `torch.float8_e8m0fnu` is absent |
| CANN | 9.0.0; actual `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0/cann-9.0.0` |
| Driver | 25.5.0 |
| Simulator compiler | `g++-15` 15.2.0 available; `gcc-15` is not resolved on PATH |

The shell's default `python` points to the separate PTOAS environment, which does
not supply the application dependencies. The old `.venv` contains another
PyPTO/simpler/Torch combination. Neither replaces `.venv-official`.
Commands in this reading edition use the application interpreter explicitly.

This venv has no `pip` module. For intentional dependency maintenance, use
`uv pip ... --python .venv-official/bin/python`; an inventory is not an install
lockfile. Package version `0.1.0` alone does not identify the compiler/runtime.
Installed PyPTO is not editable: changing its source checkout does not change
the installed package.

## Toolchain ownership

The selected PyPTO source is `/data1/home/ziruiwang/pypto-official-747e1e4b`.
Its `toolchain/versions.env` specifies PTOAS `v0.60`; its runtime submodule and
`runtime/pto_isa.pin` select simpler and PTO ISA. An unpinned clone of upstream
`main`, a shared `latest` CANN path, or an older CI pin is not this baseline.

With `PTOAS_ROOT` set, PyPTO searches that root's `ptoas`, then `ptoas.sh`, then
`bin/ptoas`, without falling back to PATH. The installed wheel environment uses
`bin/ptoas`. simpler resolves its managed ISA under the installed
`simpler_setup/_assets/build/pto-isa`; an ambient `PTO_ISA_ROOT` does not override
its pin. Keep the installed runtime and its managed ISA together.

The CANN setup path above exists. Its `ASCEND_HOME_PATH` contains both
`bin/ccec` (AICore compiler) and `tools/hcc/bin/aarch64-target-linux-gnu-g++`
(AICPU compiler). The runtime was built against this installation.

## Validate the existing environment

Save the complete code in [Current-API Smoke Example](source-01_pypto_lib--examples--current-smoke.md)
as `current_smoke.py` in the repository root before running this recipe.

```bash
.venv-official/bin/python current_smoke.py --platform a2a3sim
```

This current-interface example exercises compile, runtime and numerical
validation. The repository's beginner hello-world and 10 directory examples
still contain removed Golden arguments; their old commands are not environment
acceptance checks. See [the adapted first kernel](source-01_pypto_lib--get-started--first-kernel.md).

The installed simulator works with the available `g++-15`; a clean rebuild has
separate prerequisites, including checking any `gcc-15` requirement. A functional
simulator result does not certify real-device synchronization or performance.
Real-device recipes use `a2a3` and an allocated device set. This host has no A5
hardware; `a5sim` is a distinct simulator target. MX/E8M0 paths need their own
dtype and backend checks.

## Fresh installation and troubleshooting

The unaltered installation page (`../../../raw_wiki/01_pypto_lib/get-started/installation.md`)
preserves upstream fresh-environment instructions. They describe another setup
workflow and are not commands to overwrite this host's existing `.venv`, CANN
or selected compiler. A rebuild must pin the revisions above and verify its
toolchain and compiler prerequisites independently.

| Symptom | Local check |
| --- | --- |
| Missing PyPTO/Torch | Use `.venv-official/bin/python`; inspect `pypto.__file__`. |
| Missing `golden` | Run from the PyPTO-Lib root with that root on `PYTHONPATH`. |
| Missing PTOAS | Check `/data1/home/ziruiwang/ptoas-v0.60/bin/ptoas` and `PTOAS_ROOT`. |
| Missing E8M0 dtype | Torch 2.6.0+cpu lacks it; do not infer MX support from package import. |
| Rebuild cannot resolve GCC 15 commands | Check actual compiler executables; only `g++-15` was resolved here. |
| Device runtime failure | Check the selected profile, allocated device IDs, CANN paths and runtime build provenance. |

Full evidence and the limits of this recheck are in the
archived local audit (`../../../Corpus_Archive/20261007_input_cleanup/before/01_pypto_lib/LOCAL_AUDIT.md`).
