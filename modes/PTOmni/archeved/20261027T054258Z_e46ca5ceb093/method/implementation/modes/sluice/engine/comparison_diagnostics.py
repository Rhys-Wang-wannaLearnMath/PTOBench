"""Explain refused comparisons using fixed public codes and recorded identities.

This projection does not change compatibility, statistics or action budgets.
Never echo backend exception text, identity values or filesystem locators.
"""
import re


_FIELDS = ('case_id', 'fixture_id', 'software', 'shape', 'dtype',
           'shape_dtype_contract_ref', 'tuning', 'profile_mode', 'rank_label', 'invocation',
           'capture_level', 'clock', 'core_count', 'core_types', 'core_topology', 'program')
_KNOWN = {f'{key}_{suffix}' for key in _FIELDS for suffix in ('missing', 'mismatch')}
_KNOWN.update({
    'identity_missing', 'source_missing', 'source_mismatch', 'run_missing',
    'diagnostic_protocol_mismatch', 'analysis_stale', 'analysis_semantics_mismatch',
    'capture_repair_required', 'pmu_sample_count_mismatch',
    'physical_device_identity_unavailable', 'physical_device_changed',
    'physical_device_mapping_changed', 'device_configuration_changed', 'device_identity_missing',
    'comparison_conditions_unavailable',
})


def _physical(device):
    if not isinstance(device, dict):
        return None
    value = device.get('physical_identity')
    if not isinstance(value, dict) or value.get('status') != 'verified':
        return None
    devices = value.get('devices')
    if not isinstance(devices, list) or not 1 <= len(devices) <= 64:
        return None
    rows = []
    for row in devices:
        if (not isinstance(row, dict) or any(type(row.get(k)) is not int or row[k] < 0
                for k in ('configured_device', 'physical_device'))
                or not isinstance(row.get('uuid_sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', row['uuid_sha256']) is None):
            return None
        rows.append((row['configured_device'], row['physical_device'], row['uuid_sha256']))
    if any(len({r[i] for r in rows}) != len(rows) for i in range(3)):
        return None
    return sorted(rows)


def reasons(raw, before=None, after=None):
    """Project all known conditions; an unknown backend reason stays generic."""
    result = set()
    if not isinstance(raw, (list, tuple)):
        raw = ['comparison_conditions_unavailable']
    for reason in raw:
        if reason == 'device_mismatch':
            device_codes = set()
            devices = [(i or {}).get('device') for i in (before, after)]
            physical = [_physical(d) for d in devices]
            if any(isinstance(d, dict) and d.get('platform') in ('a2a3', 'a5') for d in devices):
                if not all(p is not None for p in physical):
                    device_codes.add('physical_device_identity_unavailable')
                elif {r[2] for r in physical[0]} != {r[2] for r in physical[1]}:
                    device_codes.add('physical_device_changed')
                elif physical[0] != physical[1]:
                    device_codes.add('physical_device_mapping_changed')
            declared = [{k: v for k, v in d.items() if k != 'physical_identity'}
                        if isinstance(d, dict) else d for d in devices]
            if declared[0] != declared[1]:
                device_codes.add('device_configuration_changed')
            result.update(device_codes or {'device_configuration_changed'})
        elif reason == 'device_missing':
            result.add('device_identity_missing')
        elif isinstance(reason, str) and reason in _KNOWN:
            result.add(reason)
        else:
            result.add('comparison_conditions_unavailable')
    return sorted(result)


