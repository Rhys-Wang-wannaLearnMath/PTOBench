"""Public Optimizer task; management and sealed implementations stay private."""
from modes._shared import knowledge


def make_prompt(experiment, record, case, instructions):
    return f'''Optimize Case {experiment['case']} for this experiment.
Your initial cwd is the registered workspace worker_000. Entry: {case['module']}::{case['entry']}.
Read .benchmark_context/README.md for the Case contract, Bench CLI and result locations.
Compiler tuning is editable in workspace-root `tuning.json`; its defaults are `PYPTO + false`.
PATH and the Bench Session environment are already set.
{knowledge.notice(record['session_root'])}
{instructions}
Exploration budget: {record['budget']['max_hardware_actions']} accepted hardware actions,
shared by all workspaces and Agent conversations in this Bench Session.
Stop when you are finished or the exploration budget is exhausted. No special final message is required.
'''
