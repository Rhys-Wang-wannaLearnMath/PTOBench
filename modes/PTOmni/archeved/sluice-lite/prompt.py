"""Initial task prompt for the independent SLUICE Lite mode."""


def make_prompt(experiment, context, instructions):
    session_root = context['session_root']
    case = context['case']
    budget = context['budget']['max_hardware_actions']
    return f'''Optimize Case {experiment['case']} for this experiment.
Your initial cwd is the registered workspace worker_000. Entry: {case['module']}::{case['entry']}.
Read .benchmark_context/README.md for the Case contract, public Bench commands and result locations.
PATH and the Bench Session environment are already set.

Read the SLUICE Lite resource guide at `{session_root}/method/README.md` for knowledge search and profile output locations.

{instructions}
Exploration budget: {budget} accepted hardware actions, shared by all workspaces and Agent conversations in this Bench Session.
Stop when finished or when the exploration budget is exhausted.
'''
