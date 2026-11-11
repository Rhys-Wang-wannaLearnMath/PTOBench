#!/bin/bash
# Run a command inside the clean-room sandbox (bubblewrap): no network except the Anthropic allowlist proxy, and a
# filesystem with only whitelisted paths (system dirs, CUDA, the task repository, EAT, the allowed reference sources).
# Anything not listed (post-cutoff libraries, other runs' transcripts, caches, other users' disks) is not visible.
#     cd <task repository> && EAT_CLEAN=<dir> $EAT_HOME/cleanroom/sandbox.sh <command ...>
# e.g. the whole run:  ... sandbox.sh bash $EAT_HOME/agent_team/launch_stateless.sh --persistent 50 3 300
# Configuration (environment):
#   EAT_CLEAN         sandbox state: home/, tmp/, claude-config/ (the agents' transcripts), run/ (proxy socket and log)
#   EAT_ROOT          the task repository (default: the current git repository), mounted read-write
#   EAT_ALLOWED_RO    space-separated host paths mounted read-only at the same location (allowed sources, venvs)
#   EAT_GPUS          GPUs the run may use (space- or comma-separated)
#   EAT_LOCKS         gpu_run lock directory, shared with runs outside the sandbox (default: $EAT_CLEAN/locks)
#   CUDA_HOME, TORCH_CUDA_ARCH_LIST
set -euo pipefail
C=${EAT_CLEAN:?set EAT_CLEAN}
EAT_HOME=$(cd "$(dirname "$0")/.." && pwd)
EAT_ROOT=${EAT_ROOT:-$(git rev-parse --show-toplevel)}
EAT_LOCKS=${EAT_LOCKS:-$C/locks}
CLAUDE_EXE=$(readlink -f "$(command -v claude)")
GPUS=${EAT_GPUS:-0}
SOCK=$C/run/proxy.sock
mkdir -p $C/run $C/home $C/tmp $C/claude-config $C/sbin "$EAT_LOCKS"
ln -sf "$CLAUDE_EXE" $C/sbin/claude
if ! [ -S "$SOCK" ] || ! pgrep -f "[a]llowlist_proxy.py --socket $SOCK" >/dev/null; then
    nohup python3 "$EAT_HOME/cleanroom/allowlist_proxy.py" --socket $SOCK --log $C/run/proxy.log >/dev/null 2>&1 &
    for _ in $(seq 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
fi

args=(
    --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64
    --symlink usr/sbin /sbin --ro-bind /etc /etc --ro-bind /sys /sys
    --dev-bind /dev /dev --proc /proc --tmpfs /tmp --tmpfs /run
    --unshare-net --unshare-ipc --unshare-uts --die-with-parent
    --ro-bind "$EAT_HOME" "$EAT_HOME"
    --ro-bind "$(dirname "$CLAUDE_EXE")" "$(dirname "$CLAUDE_EXE")"
    --bind $C $C
    --bind "$EAT_LOCKS" "$EAT_LOCKS"
    --bind "$EAT_ROOT" "$EAT_ROOT"
    --chdir "$EAT_ROOT"
    --clearenv
    --setenv HOME $C/home --setenv USER eat --setenv LANG C.UTF-8 --setenv TERM xterm
    --setenv PATH "$C/sbin:$EAT_HOME/ops:${CUDA_HOME:-/usr/local/cuda}/bin:/usr/bin:/bin"
    --setenv CUDA_HOME "${CUDA_HOME:-/usr/local/cuda}" --setenv LD_LIBRARY_PATH "${CUDA_HOME:-/usr/local/cuda}/lib64"
    --setenv TMPDIR $C/tmp --setenv XDG_CACHE_HOME $C/home/.cache
    --setenv TRITON_CACHE_DIR $C/home/.cache/triton --setenv TORCH_EXTENSIONS_DIR $C/home/.cache/torch_extensions
    --setenv CUTE_DSL_CACHE_DIR $C/home/.cache/cute_dsl
    --setenv TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST:-10.0a}" --setenv MAX_JOBS 32
    --setenv CLAUDE_CONFIG_DIR $C/claude-config --setenv CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC 1
    --setenv HTTPS_PROXY http://127.0.0.1:18080 --setenv HTTP_PROXY http://127.0.0.1:18080
    --setenv https_proxy http://127.0.0.1:18080 --setenv http_proxy http://127.0.0.1:18080
    --setenv NO_PROXY localhost,127.0.0.1 --setenv no_proxy localhost,127.0.0.1
    --setenv EAT_HOME "$EAT_HOME" --setenv EAT_ROOT "$EAT_ROOT" --setenv EAT_GPUS "${GPUS//,/ }"
    --setenv GPU_GUARD_GPUS "${GPUS// /,}" --setenv GPU_GUARD_LOCK_DIR "$EAT_LOCKS"
)
[ -d /run/nvidia-persistenced ] && args+=(--ro-bind /run/nvidia-persistenced /run/nvidia-persistenced)
for p in ${EAT_ALLOWED_RO:-}; do args+=(--ro-bind "$p" "$p"); done
for v in EAT_TASK RESUME_SESSION_ID EVAL_CMD SUPERVISOR_MODEL AGENT_MODELS SUPERVISOR_REFRESH_COST EAT_PAUSE_FILE \
         AGENT_TIMEOUT_ENFORCE CLAUDE_EXTRA_DISALLOWED CLAUDE_CODE_OAUTH_TOKEN; do
    [ -n "${!v:-}" ] && args+=(--setenv "$v" "${!v}")
done
exec bwrap "${args[@]}" bash -c 'python3 '"$EAT_HOME"'/cleanroom/tcp2unix.py 18080 '"$SOCK"' </dev/null >/dev/null 2>&1 &
bridge=$!; sleep 0.3; "$@"; rc=$?; kill $bridge 2>/dev/null; exit $rc' sandbox "$@"
