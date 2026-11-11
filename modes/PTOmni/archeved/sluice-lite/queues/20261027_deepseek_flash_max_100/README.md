# SLUICE Lite DeepSeek Flash Max queue

- tmux session: `20261027_deepseek_flash_max_100`
- mode/backend/profile: `sluice-lite` / Pi / `deepseek-flash-max` (reasoning `max`)
- order: `manifest.json`
- hardware budget: 100 exploration actions per Case; PTOBench's separate final verification action is also enabled
- serial execution: one Case at a time
- experiment configs: `configs/`
- queue status: `state.json`
- combined output: `queue.log`
- per-Case logs: `logs/`

Attach with `tmux attach -t 20261027_deepseek_flash_max_100`. Inspect status with `cat /data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench/modes/sluice-lite/queues/20261027_deepseek_flash_max_100/state.json`.
