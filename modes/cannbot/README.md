# CANNBot mode

CANNBot is a PTOBench optimization mode that uses the shared Case contract,
workspace, budget and CLI. The current supported PTOBench entry point uses Pi:

```bash
python pto-launcher run examples/cannbot_pi.yaml
```

The mode preserves its pinned upstream PyPTO performance skills and roles. Its
Pi integration loads a Session-local extension and communicates with the
Session's tool service; it does not install global Pi plugins or modify host
configuration. The public `.benchmark_context/` files and PTOBench CLI remain
the source of the Case contract and evaluation feedback.

Use `ptobench evaluate`, `ptobench profile` and `ptobench fork` for optimization
work. Hardware-action budget: 1 per accepted `ptobench evaluate` or
`ptobench profile`; a failed or interrupted accepted action still consumes one.
Fork costs 0 and is recorded as a Session operation.
When the budget reaches zero, the Launcher ends the optimization run after the current
action; further Evaluate, Profile and Fork requests are rejected.

The pinned upstream snapshot and its integrity manifest are in
[`upstream/`](upstream/) and [`upstream.lock.json`](upstream.lock.json).
