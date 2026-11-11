# PTO Wiki workflow

This offline three-stage Codex workflow reorganizes only the files listed in
`../clean_wiki/INPUT_MANIFEST.json` into the standalone `../opt_wiki/` knowledge
base. It does not call PTOBench or run NPU workloads.

## Run

Fill `PTO_WIKI_API_KEY` in the ignored `.env`, then run from the PTOBench root:

```sh
python pto_wiki/pto_wiki_pipeline/workflow.py check
python pto_wiki/pto_wiki_pipeline/workflow.py run
```

The configured agent is Codex CLI with `gpt-6.1-sol`, high reasoning, and
`https://api.aixuanhub.me/v1`. Its tool sandbox has network access disabled;
Codex CLI can still call the configured model endpoint. The workflow does not
enable web search and its prompts restrict source material to manifest-listed
files. The API key is parsed as a literal value and is never written to
workflow artifacts.

## Workflow

1. **Map:** group sources into a small set of topics and identify exact duplicate
   candidates. A local check requires complete manifest coverage.
2. **Organize:** create an `opt_wiki` candidate, preserving source explanations,
   code, tables, and distinctions between versions or contexts. Local checks
   validate the output paths and source map.
3. **Review:** a separate Codex call checks coverage, preservation, claims, and
   navigation. If it finds issues, the organizer may revise twice; a remaining
   finding stops publication.

Every stage rechecks the input hashes. A changed input stops the workflow. The
candidate and intermediate responses stay in ignored `.work/<run-id>/`. Only a
passing review replaces `opt_wiki`; the previous version is retained in that
run's `previous_opt_wiki/` directory. Mapping, source map, review, and
configuration metadata are retained under `provenance/<run-id>/`.

Stage instructions are in `prompts/`; content rules and data boundaries are in
`design/README.md`.
