"""Public Optimizer task; management and sealed implementations stay private."""

def make_prompt(experiment, record, case, instructions):
    return f'''Optimize Case {experiment['case']} for this experiment.
Your initial cwd is the registered workspace worker_000. Entry: {case['module']}::{case['entry']}.
Read .benchmark_context/README.md for the Case contract, Bench CLI and result locations.
Read .benchmark_context/tuning_capabilities.json for this Case's editable compiler and runtime scheduling options, allowed values and restrictions.
Edit Candidate Python files under `candidate/` and supported options in `candidate/tuning.json`;
the compiler defaults remain `PYPTO + false`, and the complete defaults vary by Case.
After changing source or settings, use `ptobench evaluate .` to check correctness and clean latency.
PATH and the Bench Session environment are already set.
{instructions}
Evaluate allowance: {record['budget']['max_hardware_actions']} accepted `ptobench evaluate` actions,
shared by all workspaces and Agent conversations in this Bench Session. A failed
or interrupted accepted Evaluate still consumes one. Profile budget cost is 0;
Profile is recorded as a hardware action and remains available after the Evaluate
allowance is exhausted. Fork budget cost is 0 and is recorded as a Session
operation. Extra Evaluate requests are rejected by the CLI; continue using
Profile/Fork while the Session remains open, until this Agent exits or is interrupted.
Stop when you are finished. No special final message is required.
'''
