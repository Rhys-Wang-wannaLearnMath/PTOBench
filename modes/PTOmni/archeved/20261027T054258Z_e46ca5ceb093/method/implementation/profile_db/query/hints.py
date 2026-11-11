# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bounded, ordered native compiler messages; no diagnosis or source inference."""
import re

from profile_db.facts import Evidence, Fact

_PRIVATE = re.compile(
    r'(?<!\w)(?:[A-Za-z]:[\\/]|\.{0,2}/)[^\s,;\)\]]+|'
    r'\b[\w.-]+(?:[/\\][\w.-]+)+|'
    r'sk-[A-Za-z0-9]+|api[_-]?key\s*[=:]\s*\S+|bearer\s+\S+|'
    r'hidden|sealed|oracle|evaluator|traceback|duckdb|sha256|'
    r'\b(?:action_id|event_seq|run_id|artifact_dir|locator|records|deps|sql)\b|[0-9a-f]{32,}', re.I)


def details(conn, params):
    found = []
    for seq, text, source in conn.execute('SELECT seq,text,source_path FROM perf_hint WHERE run_id=? ORDER BY seq', [params.run_id]).fetchall():
        location = re.fullmatch(r'(.*):(\d+):(\d+)', source or '')
        path = location.group(1) if location else None
        # File identity is retained separately; absolute paths never enter messages.
        raw = text[:-len(' at ' + source)] if source and text.endswith(' at ' + source) else text
        message = _PRIVATE.sub('[redacted]', raw)
        code = re.search(r'\bPH-(?:[A-Z0-9]+-)*[A-Z0-9]+\b', text)
        chunks = [message[i:i+240] for i in range(0, len(message), 240)] or ['']
        for index, chunk in enumerate(chunks):
            found.append(Fact('DETAIL', dict(entity=f'{seq}:{index}', hint_ordinal=seq,
                hint_code=code.group() if code else None, message=chunk,
                fragment_index=index, fragment_count=len(chunks), redacted=message != raw,
                source_file=path.replace('\\', '/').rsplit('/', 1)[-1] if path else None,
                source_identity=path, location_kind='native_reported',
                line=int(location.group(2)) if location else None,
                column=int(location.group(3)) if location else None), Evidence.MEASURED))
    selected = found[params.offset:params.offset+params.limit]
    return selected + [Fact('PAGE', dict(has_more=len(found)>params.offset+len(selected),
                        next_offset=params.offset+len(selected)), Evidence.PROVEN)]
