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
Exploration budget: {record['budget']['max_hardware_actions']} accepted hardware actions,
shared by all workspaces and Agent conversations in this Bench Session.
Stop when you are finished or the exploration budget is exhausted. No special final message is required.
'''
