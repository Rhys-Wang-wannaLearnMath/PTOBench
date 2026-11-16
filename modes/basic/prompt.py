"""Public Optimizer task; management and sealed implementations stay private."""

def make_prompt(experiment, record, case, instructions):
    return f'''Optimize Case {experiment['case']} for this experiment.
Your initial cwd is the registered workspace worker_000. Entry: {case['module']}::{case['entry']}.
Read .benchmark_context/README.md for the Case contract, editable options, CLI commands and result locations.
Edit Candidate Python files under `candidate/` and supported options in `candidate/tuning.json`.
After changing source or settings, use `ptobench evaluate .` to check correctness and clean latency.
The `ptobench` command is available on PATH and operates on this Session's registered workspace.
{instructions}
Hardware-action budget: {record['budget']['max_hardware_actions']} accepted `ptobench evaluate` or `ptobench profile` actions,
shared by all workspaces and Agent conversations in this Bench Session. A failed
or interrupted accepted action still consumes one; Fork does not consume it.
Stop when `remaining_hardware_actions` reaches 0; the
Launcher ends the optimization run after the current action completes, and all further
Evaluate, Profile and Fork requests are rejected.
Stop when you are finished. No special final message is required.
'''
