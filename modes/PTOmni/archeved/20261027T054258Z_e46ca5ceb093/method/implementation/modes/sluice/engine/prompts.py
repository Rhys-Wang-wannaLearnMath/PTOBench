"""Small static startup prompt; current state arrives through the Pi context hook."""


def main_prompt(context, initial=None, *, stage01_smoke_acceptance=False):
    if type(stage01_smoke_acceptance) is not bool:
        raise ValueError('stage01_smoke_acceptance must be a boolean')
    smoke_contract = """
For this Stage 01 smoke acceptance run, do not stop after a baseline diagnosis alone: reread Progress and submit the
next Decision. Complete two full optimize cycles when the available budget permits.
An optimize cycle counts only after its registered Candidate is evaluated with an integrity-passed correctness PASS
and the resulting Progress feedback is handled by a new Decision; a compile failure or `action_failed` is not a
completed cycle. Continue with a corrected, evidence-based Candidate while budget permits. A regression with valid
correctness feedback still counts as an evaluated cycle. After the first profile action, list raw files and read one
file completely (continue by offset until complete). Save its exact `profile_ref`, `file_ref`, and content digest.
After a later profile action, read that same file again with the same two references and compare the complete-content
digest. Do not call finish until both optimize cycles, a clean evaluate, a profile query, and the raw-file comparison
are verified; if budget or another constraint prevents them, explicitly report incomplete.""" if stage01_smoke_acceptance else ""
    return f"""Optimize PTOBench Case {context.case_id} while preserving its fixed public contract.
Workspace: {context.workspace}
Entry: {context.case_module}::{context.case_entry}

Read `.benchmark_context/README.md` and the other public workspace contract files before choosing a
change. The trusted SLUICE context is refreshed before every model request; use that current view
and the registered tools as the authority for state, program versions, opportunities, evidence,
knowledge, budget and next action. Do not rely on a stale view from earlier conversation turns.

Use only SLUICE tools for optimization actions. The controller sends hardware work through the
PTOBench public action interface. Do not run hardware commands directly, edit protected records,
or alter the Case contract, devices, runtime configuration or unauthorized tuning values. Keep
Candidate changes within the scope registered for the active Hypothesis. PTOBench owns finalization.
{smoke_contract}
Before you exit after meeting all requirements, submit an explicit stop Decision and call `sluice_finish`;
PTOBench performs final scoring and collection after that request.
Never read sealed assets, hidden output or benchmark control directories.
"""
