# Stage 1: map the source material

Use only `{{CLEAN_WIKI}}/INPUT_MANIFEST.json` and files listed under its
`input_files` field. Ignore `navigation_only`, `source_snapshot_path`, archived
trees, other repository files, and external sources. Do not use web search or
network access. Do not edit any file.

Group source documents into a small set of clear topics based on what a PyPTO
programmer would look up. Identify duplicate candidates only when the content is
clearly identical; list uncertain overlaps separately. Do not decide that two
documents are duplicates from their titles alone. Every `input_files[].path`
must appear in at least one topic.

Return only one raw JSON object with this shape. The runner saves the response as
`mapping.json` and checks the source coverage.

```json
{
  "schema_version": 1,
  "manifest_sha256": "{{MANIFEST_SHA256}}",
  "topics": [
    {"id": "lowercase-hyphenated-id", "title": "Topic title", "sources": ["path/from/manifest.md"]}
  ],
  "duplicate_candidates": [
    {"sources": ["path/a.md", "path/b.md"], "note": "Why the content is identical"}
  ],
  "uncertain_overlaps": [
    {"sources": ["path/a.md", "path/b.md"], "note": "What needs separate preservation"}
  ]
}
```
