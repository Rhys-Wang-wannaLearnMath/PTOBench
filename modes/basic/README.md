# Basic mode

Basic prepares its own task prompt, launches the configured native coding agent
and records its complete session. Candidate optimization choices belong to the agent.

`knowledge/raw_wiki/` contains an unchanged snapshot of PyPTO-Lib, PyPTO and
Ascend 910B documentation. `knowledge/knowledge.lock.json` pins the original
manifest and complete content fingerprint, identical to CANNBot's reference bundle.
The mode validates this pin before starting a Run and independently copies the
bundle into `SESSION/method/resources/raw_wiki/` during `prepare`.

Files are set to `0444`, directories to `0555`, and the existing participant
archive protocol registers content hashes and recovery backups. This provides
read-only permissions and integrity detection/recovery, not OS sandbox isolation.
Forks share the Session's single reference copy.

The startup prompt points to the Bench's `.benchmark_context/README.md` and the
method's `SESSION/method/README.md`. `method/knowledge.json` records the exposed
snapshot path, source/version and fingerprints. The evaluator neither copies
nor interprets these documents.
