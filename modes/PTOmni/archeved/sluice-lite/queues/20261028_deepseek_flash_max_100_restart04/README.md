# SLUICE Lite DeepSeek Flash Max queue

- tmux session: `20261028_deepseek_flash_max_100_restart04`
- mode/backend/profile: `sluice-lite` / Pi / `deepseek-flash-max` (reasoning `max`)
- order: `manifest.json`; both synthetic Pro end-to-end cases run first
- hardware budget: 100 exploration actions per Case; PTOBench's separate final verification action is enabled
- execution: serial, one Case at a time
- status: `state.json`; output: `queue.log` and `logs/`

Attach with `tmux attach -t 20261028_deepseek_flash_max_100_restart04`.
