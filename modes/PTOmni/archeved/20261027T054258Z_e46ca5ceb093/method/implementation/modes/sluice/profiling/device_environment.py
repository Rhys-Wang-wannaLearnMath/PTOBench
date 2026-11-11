"""Translate host device selection into a restricted worker environment.

The trusted caller must first resolve configured IDs to physical devices and
restrict the filesystem view to precisely those device nodes. CANN enumerates
that smaller view again. Reapplying the host's visibility indexes can select a
nonexistent index there, or select the wrong member of a distributed set.
"""


def isolated_environment(environment, mapping):
    result = dict(environment)
    if mapping is None:
        return result
    if not isinstance(mapping, dict):
        raise ValueError('isolated_device_mapping_invalid')
    configured, physical = (mapping.get(key) for key in ('configured', 'physical'))
    if (not isinstance(configured, list) or not isinstance(physical, list)
            or not configured or len(configured) != len(physical)
            or any(type(v) is not int or v < 0 for v in configured + physical)
            or len(set(configured)) != len(configured) or len(set(physical)) != len(physical)):
        raise ValueError('isolated_device_mapping_invalid')
    # Node mounts enforce the selection. configure_devices resolves each
    # selected physical ID back to its logical ID inside this restricted view.
    result.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    return result
