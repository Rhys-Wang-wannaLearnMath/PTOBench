"""Public Optimizer task; management and sealed implementations stay private."""


def make_prompt(experiment, record, case):
    return f'''Optimize Case {experiment.case} for this experiment.
Your initial cwd is the registered workspace worker_000. Entry: {case['module']}::{case['entry']}.
Read .benchmark_context/README.md and the public contract before editing.
First run ptobench evaluate to establish a baseline, then improve the Candidate.
You may edit Candidate source and helper files in registered workspaces.
Do not change .benchmark_context or Bench control records. Do not read sealed assets.
Use only these five Bench operations (PATH and Session environment are already set):
  ptobench evaluate [workspace]
  ptobench profile runtime [workspace]
  ptobench profile pmu [workspace]
  ptobench profile compiler [workspace]
  ptobench fork [workspace]
Omitting workspace uses cwd. A fork registers a new workspace; use its returned path.
Exploration budget: {record['budget']['max_hardware_actions']} accepted hardware actions,
shared by evaluate and all profile operations. fork consumes no hardware action.
Profiles provide diagnostics; their output is not an official final score.
Use these commands for Bench actions; directly invoking internal scripts violates the experiment rules.
Stop when you are finished or the exploration budget is exhausted. No special final message is required.
Bench automatically selects the best valid historical Snapshot and independently verifies it
with the one reserved finish action. You do not call finalization yourself.
'''
