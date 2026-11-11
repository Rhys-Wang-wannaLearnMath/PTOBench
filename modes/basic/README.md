# Basic mode

Basic launches the selected coding agent with the public Case contract, installed
PyPTO environment, and normal project files visible in its workspace. PTOBench
does not add a separate `raw_wiki` bundle to a Basic Session.

Pi reads its provider settings from this mode's `.env`; the checked-in
`.env.example` is the template. Basic currently uses `deepseek-flash` with
`high` reasoning. The model API endpoint is the only external host reachable
from the Agent sandbox. The default `ptobench profile [WORKSPACE]` collects
runtime, PMU and compiler diagnostics in one action. Candidate evaluation and
NPU execution use the existing PTOBench worker path.
