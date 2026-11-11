# Settings of a run, shared by the ops scripts: source it from inside the task repository.
export EAT_HOME="${EAT_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # this EAT checkout
export EAT_ROOT="${EAT_ROOT:-$(git rev-parse --show-toplevel)}"                    # the task repository
[ -f "$EAT_ROOT/eat/env.sh" ] && source "$EAT_ROOT/eat/env.sh"                     # the task's environment
export EAT_TASK="${EAT_TASK:-$(basename "$EAT_ROOT")}"
export EAT_WORK="${EAT_WORK:-$EAT_ROOT/.work}"          # caches, verification output
export EAT_LOG="${EAT_LOG:-$EAT_WORK/eat.log}"           # tee target of launch_stateless.sh
export EAT_SESSION="${EAT_SESSION:-eat_$EAT_TASK}"       # tmux session of the run
mkdir -p "$EAT_WORK"
