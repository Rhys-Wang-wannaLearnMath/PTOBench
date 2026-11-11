#!/bin/bash
# Install a task from the EAT Gym (Hugging Face dataset $EAT_GYM, default eva98/eat-gym; Harbor task format) as its own
# git repository:
#     bash ops/fetch_task.sh <task> [<dir>] [--setup] [--revision <rev>] [--from <local eat-gym dir>]
# Copies the task's environment/workspace/ (the start state, with eat/task.md, eat/eval.sh and eat/env.sh) to <dir>
# (default ./<task>). --setup also runs environment/setup.sh (venv, reference repositories, baselines, git repository
# with the start state); without it the command to run is printed.
set -euo pipefail
TASK="${1:?usage: fetch_task.sh <task> [<dir>] [--setup] [--revision <rev>] [--from <dir>]}"; shift
DEST="" REV=main FROM="" SETUP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --setup) SETUP=1; shift ;;
        --revision) REV="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        -*) echo "unknown argument $1" >&2; exit 2 ;;
        *) DEST="$1"; shift ;;
    esac
done
DEST="${DEST:-$PWD/$TASK}"
GYM="${EAT_GYM:-eva98/eat-gym}"

if [ -z "$FROM" ]; then
    FROM="${XDG_CACHE_HOME:-$HOME/.cache}/eat-gym"
    command -v hf >/dev/null || { echo "needs the Hugging Face CLI: pip install -U huggingface_hub" >&2; exit 1; }
    hf download "$GYM" --repo-type dataset --revision "$REV" --include "$TASK/*" --local-dir "$FROM" >/dev/null
fi
SRC="$FROM/$TASK"
[ -f "$SRC/task.toml" ] || { echo "no task '$TASK' in $FROM" >&2; exit 1; }
[ ! -e "$DEST" ] || [ -z "$(ls -A "$DEST")" ] || { echo "$DEST exists and is not empty" >&2; exit 1; }

mkdir -p "$DEST"
cp -R "$SRC/environment/workspace/." "$DEST/"
(cd "$SRC/environment/workspace" && find . -type f \( -name '*.sh' -o -name '*.py' \) -exec chmod +x "$DEST/{}" \;)
if [ -f "$SRC/environment/gpu_run" ]; then
    mkdir -p "$DEST/.work/bin" && cp "$SRC/environment/gpu_run" "$DEST/.work/bin/" && chmod +x "$DEST/.work/bin/gpu_run"
fi
python3 - "$SRC/task.toml" "$DEST" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
get = lambda k: (re.search(rf'^{k} = "(.*)"$', t, re.M) or [None, "?"])[1]
print(f"installed {get('name').split('/')[-1]} in {sys.argv[2]}: {get('description')}")
print(f"  hardware: {get('hardware')}")
print(f"  sources:  {get('readable_sources')}")
PY
echo "  environment: $SRC/environment/README.md"
if [ "$SETUP" = 1 ]; then
    (cd "$DEST" && source eat/env.sh && bash "$SRC/environment/setup.sh")
else
    echo "  set up with: (cd $DEST && source eat/env.sh && bash $SRC/environment/setup.sh)   (or rerun with --setup)"
fi
