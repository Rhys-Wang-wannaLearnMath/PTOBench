#!/bin/bash
# Wait for the clean run's iteration N to be scored, then re-run the trace and code-overlap audits.
#     cleanroom/audit_after_iter.sh N
# Environment: as for audit_clean_traces.py, plus EAT_LOG (the run log; default <task repository>/.work/eat.log),
# EAT_ALLOWED_SRC and EAT_FORBIDDEN_SRC (comma-separated source dirs for audit_code_overlap.py) and EAT_CANDIDATE_SRC
# (the task's solution files or dirs, comma-separated; default: the task repository).
N=${1:?iteration}
D=$(cd "$(dirname "$0")" && pwd)
EAT_ROOT=${EAT_ROOT:-$(git rev-parse --show-toplevel)}; export EAT_ROOT
EAT_LOG=${EAT_LOG:-$EAT_ROOT/.work/eat.log}
EAT_CANDIDATE_SRC=${EAT_CANDIDATE_SRC:-$EAT_ROOT}
OUT=$EAT_CLEAN/run/audit_iter$N.txt
until grep -q "Iteration $N complete" "$EAT_LOG"; do
    grep -q -E "budget exhausted|SUPERVISOR EXPERIMENT COMPLETE" "$EAT_LOG" && break
    sleep 120
done
{
    date -u
    grep -E "iter $N: .*score=" "$EAT_LOG" | tail -1
    python3 "$D/audit_clean_traces.py" "$EAT_CLEAN/run/clean_trace_audit_iter$N.txt"
    python3 "$D/audit_code_overlap.py" "$EAT_CANDIDATE_SRC" "$EAT_ALLOWED_SRC" "$EAT_FORBIDDEN_SRC"
    cat "$EAT_CLEAN/run/proxy.log"
    echo DONE
} > "$OUT" 2>&1
