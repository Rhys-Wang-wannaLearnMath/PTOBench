# Stage 2: create the standalone pypto_wiki

This is an already-authorized, unattended workflow run. Carry out the requested
organization and return the required artifacts directly; no approval or
clarification is needed.

Use only files listed in `{{CLEAN_WIKI}}/INPUT_MANIFEST.json` and the mapping at
`{{MAPPING}}`. Ignore `navigation_only`, `source_snapshot_path`, archived trees,
other repository files, and external sources. Do not use web search or network
access. Do not edit `clean_wiki` or any file outside `{{CANDIDATE}}`.

If `{{REVIEW_FEEDBACK}}` is non-empty, fix only the listed review findings
while preserving all existing candidate content that is not implicated. Recheck
the complete source map and links after the revision.

Create a self-contained knowledge base in the current directory (`{{CANDIDATE}}`):

- `README.md`: purpose and how to use the index.
- `INDEX.md`: a short table mapping common questions to topic pages/documents.
- `topics/<topic-id>/README.md`: brief navigation for that topic.
- `topics/<topic-id>/*.md`: organized source content.
- `source-map.json`: maintenance metadata described below. The runner moves it
  out of the candidate after review, so it is not part of PTOmni's knowledge.

Keep this a conservative reorganization. Preserve source explanations, code,
tables, headings, and technical terms. Group related documents; merge only
clearly identical passages in the same context. Keep similar but distinct,
version-specific, hardware-specific, or conflicting material separate. Do not
write new optimization advice, rankings, risk assessments, performance claims,
or technical conclusions. Update only titles, navigation, and relative links
needed by the new layout. Do not link back to `clean_wiki` or the network.

Map every manifest input to one or more output documents. When content is merged,
record every contributing source. Use exactly this JSON structure in
`source-map.json`:

```json
{
  "schema_version": 1,
  "manifest_sha256": "{{MANIFEST_SHA256}}",
  "sources": [
    {"path": "path/from/manifest.md", "outputs": ["topics/topic-id/document.md"]}
  ]
}
```

Every output path must be relative to the current directory and point to a file
created in the candidate. Include each manifest input exactly once in `sources`.
