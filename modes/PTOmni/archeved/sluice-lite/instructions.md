Optimize the public Candidate for lower latency while preserving its contract and correctness.

Use the Case instructions and `ptobench` commands from `.benchmark_context/README.md`. Run each `ptobench` action in the foreground and wait for its completed result before continuing; do not launch actions with `nohup`, `&`, or another background process. Run `ptobench evaluate .` after each meaningful source change, and run it at least once before stopping even if you leave the Candidate unchanged; PTOBench needs a successful clean evaluation to finish with a valid result. Use `ptobench profile runtime .` or another mode listed in `.benchmark_context/profiling_modes.json` when diagnostics can answer a concrete question.

The structured knowledge package is read-only at `$PTOBENCH_SESSION_ROOT/method/resources/knowledge`. Search `index/search.json`, then read the relevant card under `cards/` and any needed `reference/` entry. Treat cards as documented mechanisms and constraints, not guarantees that a change will improve this Candidate.

PTOBench profile artifacts are available under `.benchmark_outputs/<mode>/` and may be inspected directly when a diagnostic question needs them. Each `ptobench profile ...` command consumes a hardware action.

Change Candidate Python source only. Do not edit `.benchmark_context/`, hidden benchmark assets, or the knowledge snapshot. No structured hypothesis, decision report, or final message is required.
