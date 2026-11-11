# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Load anchored JSON or static compiler kernel identities without executing code."""
import json
from pathlib import Path
from profile_db.errors import IngestError


def load(name_map):
    name_map = Path(name_map)
    if name_map.suffix == '.py':
        import ast
        try:
            tree = ast.parse(name_map.read_text())
            assignments = [n.value for n in tree.body if isinstance(n,ast.Assign)
                           and any(isinstance(t,ast.Name) and t.id=='KERNELS' for t in n.targets)]
            if len(assignments)!=1 or not isinstance(assignments[0],ast.List):
                raise ValueError('KERNELS list missing')
            mapping = {}
            for node in assignments[0].elts:
                if not isinstance(node,ast.Dict):
                    raise ValueError('kernel entry is not a dictionary')
                fields = {k.value:v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant)}
                number = ast.literal_eval(fields['func_id'])
                name = ast.literal_eval(fields['name'])
                if type(number) is not int or number<0 or not isinstance(name,str) or not name or str(number) in mapping:
                    raise ValueError('invalid or duplicate kernel identity')
                mapping[str(number)] = name
            if not mapping:
                raise ValueError('empty kernel mapping')
            return mapping
        except (ValueError,KeyError,SyntaxError,TypeError) as exc:
            raise IngestError('invalid dispatch kernel configuration') from exc
    data = json.loads(name_map.read_text(encoding="utf-8"))
    mapping = data.get("callable_id_to_name")
    if not isinstance(mapping, dict):
        raise IngestError(f"{name_map}: missing callable_id_to_name mapping")
    return {str(key): str(value) for key, value in mapping.items()}



def document(name_map):
    if Path(name_map).suffix != '.py':
        return json.loads(Path(name_map).read_text())
    return {"callable_id_to_name":load(name_map)}
