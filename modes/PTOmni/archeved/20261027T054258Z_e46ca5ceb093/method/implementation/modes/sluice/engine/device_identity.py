"""Bounded public physical identity navigation from retained parent originals."""
from .evidence_guard import encoded, opaque, require_ref, scan


def read_identity(db, handle):
    from profile_db.runtime_adapter.storage import resolve
    from ..profiling.device_identity import comparison_identity, unavailable, validate
    from .profile_adapter import read_json
    device = handle.identity.device or {}
    if device.get('platform') not in ('a2a3', 'a5'):
        return None
    expected = device.get('physical_identity')
    if expected is None:
        return unavailable('legacy_identity_missing')
    kind = 'clean_result' if handle.identity.profile_mode == 'clean' else 'device_identity_result'
    present = db.connection.execute('SELECT 1 FROM artifact WHERE run_id=? AND kind=?',
                                    [handle.db_run_id, kind]).fetchone()
    if not present and expected.get('status') != 'verified':
        return unavailable(expected.get('reason', 'legacy_identity_missing'))
    try:
        raw = read_json(resolve(db.connection, handle.db_run_id, kind))
        rows = [r for r in raw['runs'] if r.get('fixture_id') == handle.identity.fixture_id]
        if len(rows) != 1:
            raise ValueError('device_identity_original_invalid')
        value = rows[0].get('device_identity')
        if comparison_identity(value, handle.action_id) != expected:
            raise ValueError('device_identity_original_invalid')
        return unavailable('legacy_identity_missing') if value is None else validate(value)
    except Exception as exc:
        raise ValueError('device_identity_original_invalid') from exc


def details(adapter, handle, cursor, budget_bytes):
    from .controller import save
    from .profile_adapter import read_json
    if type(budget_bytes) is not int or not 1024 <= budget_bytes <= 4096:
        raise ValueError('evidence_budget_invalid')
    if handle.member_kind == 'fixture':
        raise ValueError('profile_detail_requires_member_ref')
    leaf = adapter.handles[handle.member_refs[0]] if handle.member_refs else handle
    doc = read_identity(adapter.db, leaf)
    binding = opaque('evidence', encoded(
        [handle.ref, handle.artifact_fingerprint, 'physical-device-v1']).decode())
    root = adapter.context.work_root / 'profile_cursors'
    offset = 0
    if cursor is not None:
        require_ref(cursor, 'evidence')
        saved = read_json(root / (cursor.split(':')[1] + '.json'))
        if (saved.get('binding') != binding or type(saved.get('offset')) is not int
                or saved['offset'] < 0):
            raise ValueError('detail_cursor_mismatch')
        offset = saved['offset']
    devices = doc['devices'] if doc and doc['status'] == 'verified' else []
    if 'rank' in (handle.identity.device or {}):
        devices = [d for d in devices if d['configured_device'] == handle.identity.device['device']]
    if offset and offset >= len(devices):
        raise ValueError('detail_cursor_mismatch')
    items = []
    for index, device in enumerate(devices[offset:offset+8], offset):
        fields = {'configured_device': device['configured_device'],
                  'physical_device_ref': opaque('evidence', ('npu-uuid', device['uuid_sha256'])),
                  'pci_status': device['pci']['status']}
        if device['pci']['status'] == 'verified':
            fields['pci_ref'] = opaque('evidence', ('npu-pci', device['uuid_sha256'], device['pci']['bdf']))
        items.append({'entity_ref': opaque('evidence', (binding, index)),
                      'status': 'measured', 'fields': fields})
    while True:
        following = offset + len(items)
        more = following < len(devices)
        next_cursor = opaque('evidence', (binding, following)) if more else None
        value = {'profile_ref': handle.ref, 'view': 'device_identity',
                 'status': 'available' if devices else 'unavailable', 'items': items,
                 'reason': None if devices else doc['reason'] if doc else 'non_npu_target',
                 'device_count': len(devices), 'scope': 'parent_selected_physical_devices',
                 'global_rank_certified': False, 'next_cursor': next_cursor,
                 'truncated': more, 'evidence_ref': binding}
        scan(value)
        if len(encoded(value)) <= budget_bytes:
            if next_cursor:
                save(root / (next_cursor.split(':')[1] + '.json'),
                     {'binding': binding, 'offset': following})
            return value
        if len(items) <= 1:
            raise ValueError('evidence_budget_invalid')
        items.pop()
