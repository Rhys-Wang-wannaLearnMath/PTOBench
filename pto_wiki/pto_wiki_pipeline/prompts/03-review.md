# Stage 3: independent review

Review the candidate in `{{CANDIDATE}}` against
`{{CLEAN_WIKI}}/INPUT_MANIFEST.json`, the listed source files, the mapping at
`{{MAPPING}}`, and the source map at `{{SOURCE_MAP}}`. Do not edit any file. Do
not use web search, network access, archived trees, or other repository sources.

Check that:

1. Every manifest input is represented in the source map and candidate output;
   merged passages retain all contributing source paths.
2. Source explanations, code, tables, and distinct versions or contexts have
   been preserved. Any uncertain duplicate remains separate.
3. `README.md`, `INDEX.md`, and topic navigation work without reading
   `clean_wiki`; candidate links resolve inside the candidate.
4. No unsupported technical claims, advice, rankings, or external-source
   material were added.

Return only this raw JSON shape. Use `pass` only when all checks pass; otherwise
use `revise` and list concrete findings. Do not fix findings yourself.

```json
{"status": "pass", "findings": []}
```
