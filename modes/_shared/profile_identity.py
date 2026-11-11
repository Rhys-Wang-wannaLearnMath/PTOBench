"""Trusted, context-free NPU identity collection outside measured execution.

ACL supplies the globally unique 16-byte UUID. DCMI PCI coordinates are optional
diagnostic metadata; they never substitute for a UUID. Importing this module
does not load a driver, initialize a device or dispatch work.
"""
from __future__ import annotations

import ctypes
import hashlib
import re

from .profile_environment import isolated_environment

SCHEMA_VERSION = 1
ORIGIN = 'trusted_parent_acl'
REASONS = frozenset({
    'acl_unavailable', 'acl_mapping_query_failed', 'acl_mapping_changed',
    'acl_uuid_query_failed', 'acl_uuid_zero', 'acl_uuid_duplicate',
    'device_changed_during_action', 'identity_not_observed', 'legacy_identity_missing',
})


class _Uuid(ctypes.Structure):
    _fields_ = [('bytes', ctypes.c_ubyte * 16)]


class _Pcie(ctypes.Structure):
    # dcmi_pcie_info_all, dcmi_interface_api.h. Refuse a different host ABI.
    _fields_ = [(name, ctypes.c_uint32) for name in
                ('vendor', 'subvendor', 'device', 'subdevice')]
    _fields_ += [('domain', ctypes.c_int32)]
    _fields_ += [(name, ctypes.c_uint32) for name in ('bus', 'slot', 'function')]
    _fields_ += [('reserved', ctypes.c_ubyte * 32)]


def _function(library, name, arguments):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = ctypes.c_int32
    return function


def _pci(physical, load):
    unavailable = {'status': 'unavailable'}
    if ctypes.sizeof(_Pcie) != 64:
        return unavailable
    try:
        library = load('/usr/local/Ascend/driver/lib64/driver/libdcmi.so')
        if _function(library, 'dcmi_init', [])():
            return unavailable
        logical, card, device = ctypes.c_uint32(), ctypes.c_int32(), ctypes.c_int32()
        pointer = ctypes.POINTER
        if _function(library, 'dcmi_get_device_logicid_from_phyid',
                     [ctypes.c_uint32, pointer(ctypes.c_uint32)])(physical, ctypes.byref(logical)):
            return unavailable
        if _function(library, 'dcmi_get_card_id_device_id_from_logicid',
                     [pointer(ctypes.c_int32), pointer(ctypes.c_int32), ctypes.c_uint32])(
                         ctypes.byref(card), ctypes.byref(device), logical.value):
            return unavailable
        if card.value < 0 or device.value < 0:
            return unavailable
        value = _Pcie()
        if _function(library, 'dcmi_get_device_pcie_info_v2',
                     [ctypes.c_int32, ctypes.c_int32, pointer(_Pcie)])(
                         card.value, device.value, ctypes.byref(value)):
            return unavailable
        if not (0 <= value.domain <= 65535 and value.bus <= 255
                and value.slot <= 31 and value.function <= 7):
            return unavailable
        return {'status': 'verified',
                'bdf': f'{value.domain:04x}:{value.bus:02x}:{value.slot:02x}.{value.function}'}
    except (OSError, AttributeError, TypeError, ValueError):
        return unavailable


def unavailable(reason):
    if reason not in REASONS:
        raise ValueError('device_identity_reason_invalid')
    return {'schema_version': SCHEMA_VERSION, 'origin': ORIGIN,
            'status': 'unavailable', 'reason': reason}


def collect(mapping, *, load=ctypes.CDLL):
    """Bind exactly the parent's selected logical/physical pairs to UUID hashes.

    CPU targets return None. A valid mapping with an unsupported UUID query is
    explicitly unavailable, so independent unknown actions cannot compare equal.
    """
    if mapping is None:
        return None
    isolated_environment({}, mapping)  # Validate before touching any driver.
    try:
        library = load('libascendcl.so')
        physical_id = _function(library, 'aclrtGetPhyDevIdByLogicDevId',
                                [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)])
        uuid = _function(library, 'aclrtDeviceGetUuid',
                         [ctypes.c_int32, ctypes.POINTER(_Uuid)])
    except (OSError, AttributeError):
        return unavailable('acl_unavailable')
    devices = []
    for configured, physical in zip(mapping['configured'], mapping['physical']):
        actual, value = ctypes.c_int32(-1), _Uuid()
        if physical_id(configured, ctypes.byref(actual)):
            return unavailable('acl_mapping_query_failed')
        if actual.value != physical:
            return unavailable('acl_mapping_changed')
        if uuid(configured, ctypes.byref(value)):
            return unavailable('acl_uuid_query_failed')
        data = bytes(value.bytes)
        if not any(data):
            return unavailable('acl_uuid_zero')
        devices.append({'configured_device': configured, 'physical_device': physical,
                        'uuid_sha256': hashlib.sha256(data).hexdigest(), 'pci': _pci(physical, load)})
    if len({d['uuid_sha256'] for d in devices}) != len(devices):
        return unavailable('acl_uuid_duplicate')
    return {'schema_version': SCHEMA_VERSION, 'origin': ORIGIN,
            'status': 'verified', 'devices': devices}


def validate(value):
    """Validate saved parent metadata without querying today's hardware."""
    def require(condition):
        if not condition:
            raise ValueError('device_identity_invalid')
    require(isinstance(value, dict))
    require(type(value.get('schema_version')) is int and value['schema_version'] == SCHEMA_VERSION
            and value.get('origin') == ORIGIN)
    if value.get('status') == 'unavailable':
        require(set(value) == {'schema_version', 'origin', 'status', 'reason'}
                and isinstance(value.get('reason'), str) and value['reason'] in REASONS)
        return value
    require(value.get('status') == 'verified'
            and set(value) == {'schema_version', 'origin', 'status', 'devices'})
    devices = value.get('devices')
    require(isinstance(devices, list) and 1 <= len(devices) <= 64)
    for device in devices:
        require(isinstance(device, dict)
                and set(device) == {'configured_device', 'physical_device', 'uuid_sha256', 'pci'})
        require(all(type(device.get(k)) is int and 0 <= device[k] <= 2**31-1
                    for k in ('configured_device', 'physical_device')))
        require(isinstance(device.get('uuid_sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', device['uuid_sha256']) is not None)
        pci = device.get('pci')
        require(isinstance(pci, dict))
        require(pci == {'status': 'unavailable'} or (
            set(pci) == {'status', 'bdf'} and pci['status'] == 'verified'
            and isinstance(pci['bdf'], str)
            and re.fullmatch('[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\\.[0-7]', pci['bdf']) is not None))
    require(all(len({d[key] for d in devices}) == len(devices)
                for key in ('configured_device', 'physical_device', 'uuid_sha256')))
    return value


def _binding(value):
    return [{k: d[k] for k in ('configured_device', 'physical_device', 'uuid_sha256')}
            for d in value['devices']]


def attach(run, before, after):
    """Overwrite worker claims using only the trusted parent's observations."""
    run.pop('device_identity', None)
    if before is None and after is None:
        return
    for value in (before, after):
        if value is not None:
            validate(value)
    if before is None or after is None:
        run['device_identity'] = unavailable('identity_not_observed')
    elif before['status'] != 'verified' or after['status'] != 'verified':
        run['device_identity'] = before if before['status'] != 'verified' else after
    elif _binding(before) != _binding(after):
        run['device_identity'] = unavailable('device_changed_during_action')
    else:
        run['device_identity'] = before


def comparison_identity(value, action_id):
    """PCI availability cannot make an otherwise identical UUID incompatible."""
    value = unavailable('legacy_identity_missing') if value is None else validate(value)
    if value['status'] != 'verified':
        return {'status': 'unavailable', 'reason': value['reason'], 'action_id': action_id}
    return {'status': 'verified', 'devices': _binding(value)}


def bind_formal(identity, run, action_id):
    """Use the finalizer's own observation, never reuse exploration provenance."""
    device = identity.get('device')
    if not isinstance(device, dict) or device.get('platform') not in ('a2a3', 'a5'):
        return identity
    physical = comparison_identity(run.get('device_identity'), action_id)
    if physical['status'] != 'verified' or physical != device.get('physical_identity'):
        raise ValueError('formal_physical_device_identity_changed_or_unavailable')
    return {**identity, 'device': {**device, 'physical_identity': physical}}
