# Basic mode

Basic launches the selected coding agent with the public Case contract, installed
PyPTO environment, and normal project files visible in its workspace. PTOBench
does not add a separate `raw_wiki` bundle to a Basic Session.

Pi reads its provider settings from this mode's `.env`; the checked-in
`.env.example` is the template. Basic currently uses `deepseek-flash` with
`high` reasoning. The model API endpoint is the only external host reachable
from the Agent sandbox. The default `ptobench profile [WORKSPACE]` collects
runtime, PMU and compiler diagnostics. Evaluate budget is one per accepted
`ptobench evaluate`; a failed or interrupted accepted Evaluate still consumes
its allowance. Profile budget cost is 0; it remains recorded as a hardware
action and can run after the Evaluate allowance is exhausted. Fork budget cost
is 0 and the operation is recorded. The CLI rejects extra Evaluate requests,
while the Agent can continue Profile/Fork until the Session closes or the user
interrupts it. Candidate evaluation and NPU execution use the existing PTOBench
worker path.
