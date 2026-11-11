#!/usr/bin/env bash
set -u
REPO_ROOT="/data1/home/ziruiwang/pypto-lib"
PTOBENCH_ROOT="$REPO_ROOT/benchmarks/PTOBench"
QUEUE_DIR="$PTOBENCH_ROOT/modes/sluice-lite/queues/20261028_deepseek_flash_max_100_restart06"
PYTHON="$REPO_ROOT/.venv-official/bin/python"
export PTOBENCH_ROOT PTOBENCH_REPO_ROOT="$REPO_ROOT"
export PATH="$PTOBENCH_ROOT:$PATH"
cd "$REPO_ROOT"
"$PYTHON" -u "$QUEUE_DIR/run_queue.py"
queue_exit=$?
printf 'Queue runner exit: %s\n' "$queue_exit"
exec bash --noprofile --norc
