# Experiment Launcher

Use an interpreter with PyPTO-Lib's Torch/PyPTO dependencies for Case metadata
preflight, and install the local Launcher dependency in that same interpreter:

```bash
export PTOBENCH_ROOT=/path/to/PTOBench  # 或 pypto-lib/benchmarks/PTOBench
export PTOBENCH_REPO_ROOT=/path/to/pypto-lib
# 可选：将不放入 Git 仓库的 sealed_assets 挂载到这里
# export PTOBENCH_SEALED_ROOT=/path/to/PTOBench-assets/sealed_assets
python -m pip install -r "$PTOBENCH_ROOT/requirements.txt"
python "$PTOBENCH_ROOT/pto-launcher" run "$PTOBENCH_ROOT/examples/experiment.yaml"
python "$PTOBENCH_ROOT/pto-launcher" status /absolute/path/to/session
```

The example uses Codex. Select `codex`, `claudecode`, `pi`, or `opencode` under
`optimizer.backend` and a named profile under `optimizer.profile`. Tools use
existing host authentication. Their executables must be available on PATH.
Pi and OpenCode profiles use the pypto-lib provider launcher when that nested
integration is present; an independent PTOBench checkout automatically uses
the installed `pi` or `opencode` executable instead.
The default fallback uses the tool's own configured model and records it as
`unknown`. Named gateway profiles retain their provider/model arguments, so
the corresponding provider must be configured in the host tool. An existing
custom wrapper is required as configured and is never silently discarded.
The initial machine verification and backend readiness results are recorded in
[the S5 report](reports/agent_upgrade/S5_launcher_and_adapters.md).

```yaml
case: deepseek_v4.rmsnorm_decode_demo
optimizer:
  id: codex
  backend: codex
  profile: default
output:
  runs_root: ../runs
  snapshots_root: ../snapshots
budget:
  max_hardware_actions: 20
  finish_actions: 1
```

Case and optimizer id/backend are required. Defaults are profile `default`,
output `runs`/`snapshots`, and budget 20/1. Relative outputs resolve against the
experiment YAML directory, even when launching from another cwd. Safe YAML
mapping, duplicate keys, unknown fields, invalid IDs/budgets, missing profiles,
symlinks in managed paths, and output overlap are checked before creating a
Session. Preflight verifies prepared Case assets, the configured Python/CANN/
PTOAS installation and NPU health, installed backend/version/flags, and the
absence of retired entries. It performs no Ref preparation or measurement.

Profiles live in `launcher/profiles/<backend>/<name>.yaml`. They contain string
arrays `argv` and `version_argv`, optional `help_argv` and `model`, and a string
mapping `env`. `{prompt}` must occur as one separate argv element;
`{workspace}`, `{session_root}` and `{repo_root}` are also available. Commands run directly
with `shell=False`; shell strings are not interpreted. `env` accepts ordinary
settings, including OpenCode's permission configuration, and rejects credential
settings and overrides of managed PATH/Session variables. Keep credentials in
the tool's existing host authentication. The default model is the tool's own
model; provenance records `unknown` when the profile does not specify it.

The Launcher creates a fresh Session and `worker_000`, injects PATH and
`PTOBENCH_SESSION_ROOT`, `PTOBENCH_SESSION_ID`, `PTOBENCH_CASE_ID`,
`PTOBENCH_OPTIMIZER_ID`, `PTOBENCH_WORKSPACE`, and starts the Optimizer with that
workspace as cwd in a separate process group. The prompt presents evaluate,
profile runtime/pmu/compiler, and fork, plus source/context rules and the shared
budget. The Agent can invoke `ptobench` immediately. Its first evaluation
establishes a baseline.

Session records include protected `experiment.json`, `profile.json`,
`prompt.txt`, `backend.json` (actual argv, version, profile hash and model),
`optimizer_process.json`, and `launcher_result.json`. stdout/stderr are forwarded
and retained separately, and combined in `process.log`. Tool text is diagnostic;
official values come from the independently verified `final_result.json`.

The monitor reads Session state approximately every second. On budget exhaustion
it waits for the last accepted action to finish and clear `active_action` before
closing exploration and stopping the Optimizer. On normal/nonzero exit or
SIGINT/SIGTERM it closes exploration, stops remaining Optimizer workers and any
attributed evaluator group, reaps processes, recovers interrupted actions without
refunding budget, and invokes S4 finalization. Owned groups receive TERM, up to
10 seconds grace, then KILL if required. There is no exploration wall clock or
inactivity timeout. Infrastructure failure stops related groups and blocks final
hardware execution; unrecoverable records are retained as evidence.

S4 selects the best valid historical Snapshot, or records `no_valid_snapshot`
without consuming finish if none exists. The one finish action independently
verifies the selection. Optimizer nonzero exit can still yield final success.
Exit codes are 0 for verified success, 1 for ordinary final failure, 2 for an
invalid request, 5 for infrastructure failure, and 130 for user interruption
(including interruption during final measurement). Run prints Session/final
paths as JSON after draining logs. Status validates/restores record evidence and
shows identity, lifecycle, usage, remaining exploration budget, active action,
and any committed final result path; it invokes no hardware or finalization.

CPU subprocess tests live in `evaluator/tests/test_launcher.py`. Run them with
the same interpreter containing PyYAML. They use only temporary fake Cases,
four fake tools and a controlled Evaluator. Their results do not certify real
backend authentication or NPU measurement equivalence. See the
[S6 acceptance report](reports/agent_upgrade/S6_integration_acceptance.md) for
representative hardware measurements, per-backend readiness and raw evidence.
S7 full Case acceptance is a separate stage.

Run the complete control regression in either checkout layout with the
Torch/PyPTO interpreter used above:

```bash
python -m unittest discover -s "$PTOBENCH_ROOT/evaluator/tests" -p 'test_*.py' -v
```

Session/Launcher tests provision their own small CPU assets. The complete suite
also reads Case evaluator source and certification from `PTOBENCH_SEALED_ROOT`
for comparator checks, so provide a matching asset bundle. It invokes no NPU.
Standalone portability regressions additionally
exercise a renamed checkout, external roots and the Pi/OpenCode fallback.
