# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pure, non-executing resource extraction from AllocateMemoryAddr IR dumps.

A local allocation's size is distinct from the per-space address extent after
reuse/alignment. Hardware limits come only from anchored compiler metadata.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json

from profile_db.errors import IngestError

PARSER_VERSION = 'allocation-ir/3'


def symbol(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = symbol(node.value)
        return base + '.' + node.attr if base else None
    return None


def integer(node):
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0:
        return node.value
    if isinstance(node, ast.Call) and symbol(node.func) == 'pl.const' and node.args:
        return integer(node.args[0])
    return None


def parse(text, hardware=None):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise IngestError('invalid allocation IR') from exc
    hardware = hardware or {}
    limits = hardware.get('limits_bytes', {})
    if hardware and (hardware.get('schema_version') != 1 or not isinstance(limits, dict)):
        raise IngestError('invalid compiler hardware metadata')
    if any(not isinstance(k, str) or type(v) is not int or v <= 0 for k, v in limits.items()):
        raise IngestError('invalid compiler memory limit')
    kernels, seen = [], set()
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        allocations, views, errors = {}, defaultdict(set), []
        calls = {id(n) for n in ast.walk(fn) if isinstance(n, ast.Call) and symbol(n.func) == 'pl.tile.alloc'}
        assigned_calls = set()
        for node in ast.walk(fn):
            if isinstance(node, (ast.AnnAssign, ast.Assign)):
                call = node.value
                if not isinstance(call, ast.Call) or symbol(call.func) != 'pl.tile.alloc':
                    continue
                assigned_calls.add(id(call))
                targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
                if len(targets) != 1 or not isinstance(targets[0], ast.Name) or len(call.args) != 2 or call.keywords:
                    errors.append('unsupported_allocation')
                    continue
                name = targets[0].id
                space = symbol(call.args[0])
                size = integer(call.args[1])
                if not space or not space.startswith('pl.Mem.') or name in allocations:
                    errors.append('ambiguous_allocation')
                    continue
                allocations[name] = {'buffer': name, 'space': space.rsplit('.', 1)[-1],
                    'size_bytes': size, 'line': node.lineno}
                if size is None:
                    errors.append('dynamic_allocation')
            if isinstance(node, ast.Call) and symbol(node.func) == 'pl.MemRef' and len(node.args) >= 3:
                if isinstance(node.args[0], ast.Name):
                    views[node.args[0].id].add((integer(node.args[1]), integer(node.args[2])))
        if calls-assigned_calls:
            errors.append('unsupported_allocation')
        if not allocations and not errors:
            continue
        if fn.name in seen:
            raise IngestError('ambiguous allocation IR kernel name')
        seen.add(fn.name)
        spaces = defaultdict(list)
        for name, entry in allocations.items():
            observed = views[name]
            values = {offset for offset, _ in observed}
            # A full-size view identifies the allocation base. Smaller views
            # may start later; choosing min(offset) would hide ambiguous or
            # out-of-bounds subviews when the full view is absent.
            full = {offset for offset, size in observed if size == entry['size_bytes'] and size is not None}
            offset = next(iter(full)) if len(full) == 1 else (
                next(iter(values)) if len(values) == 1 else None)
            if offset is not None and entry['size_bytes'] is not None:
                if any(o is None or n is None or o < offset or o + n > offset + entry['size_bytes']
                       for o, n in observed):
                    errors.append('invalid_allocation_view')
                    offset = None
            entry['offset_bytes'] = offset
            if offset is None:
                errors.append('allocation_offset_unavailable')
            spaces[entry['space']].append(entry)
        # An unparsed allocation can occupy any space. Known entries remain
        # inspectable, but cannot establish a complete extent or largest buffer.
        allocations_complete = not any(e in errors for e in ('unsupported_allocation', 'ambiguous_allocation'))
        summaries = []
        for space, entries in sorted(spaces.items()):
            complete = allocations_complete and all(e['size_bytes'] is not None and e['offset_bytes'] is not None for e in entries)
            usage = max(e['offset_bytes'] + e['size_bytes'] for e in entries) if complete else None
            limit = limits.get(space)
            summaries.append({'space': space, 'usage_bytes': usage, 'limit_bytes': limit,
                'headroom_bytes': limit - usage if limit is not None and usage is not None else None,
                'largest_buffer_bytes': max((e['size_bytes'] for e in entries if e['size_bytes'] is not None), default=None),
                'allocation_count': len(entries), 'complete': complete,
                'basis': 'allocated_address_extent'})
        kernels.append({'kernel': fn.name, 'signature': hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
            'allocations': list(allocations.values()), 'spaces': summaries,
            'complete': not errors, 'allocations_complete': allocations_complete, 'errors': sorted(set(errors))})
    return {'schema_version': 1, 'parser_version': PARSER_VERSION,
        'hardware_backend': hardware.get('backend_type'), 'kernels': kernels,
        'complete': bool(kernels) and all(k['complete'] for k in kernels)}


def read(ir_path, hardware_path=None):
    hardware = json.loads(hardware_path.read_text()) if hardware_path is not None else None
    return parse(ir_path.read_text(), hardware)
