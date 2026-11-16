# Basic mode

Basic launches the selected coding agent with the public Case contract, installed
PyPTO environment, and normal project files visible in its workspace. PTOBench
does not add a separate `raw_wiki` bundle to a Basic Session.

Pi reads its provider settings from this mode's `.env`; the checked-in
`.env.example` is the template. Basic currently uses `deepseek-flash` with
`high` reasoning. The model API endpoint is the only external host reachable
from the Agent sandbox. The default `ptobench profile [WORKSPACE]` collects
runtime, PMU and compiler diagnostics. Hardware-action budget: 1 per accepted
`ptobench evaluate` or `ptobench profile`; a failed or interrupted accepted
action still consumes one. Fork costs 0 and is recorded as a Session operation. Once
the budget reaches zero, the Launcher ends the optimization run after the current action;
further Evaluate, Profile and Fork requests are rejected.
Candidate evaluation and NPU execution use the existing PTOBench worker path.
