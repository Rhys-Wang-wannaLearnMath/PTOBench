"""Optional bounded access to durable public captures by registered profile identity."""
import json
import os
import socket

from ..foundation.protocol import ProtocolError


def read(controller, profile_ref, file_ref=None, offset=0):
    context = controller.context
    config = json.loads((context.method_root / 'config.json').read_text())
    if config.get('allow_raw_profile_read') is not True:
        return {'status': 'unavailable', 'reason': 'raw_profile_access_disabled', 'hardware_actions_used': 0}
    if profile_ref not in controller.refs:
        raise ProtocolError('Unknown profile reference')
    ref = controller.refs[profile_ref]
    if ref.mode not in ('runtime', 'pmu', 'compiler'):
        raise ProtocolError('Raw access requires a diagnostic profile capture')
    request = {'event_seq': ref.event_seq, 'file_ref': file_ref, 'offset': offset}
    broker = os.environ.get('PTOBENCH_RPC_SOCKET')
    try:
        if broker:
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(120)
                connection.connect(broker)
                connection.sendall((json.dumps({'profile_archive': request}) + '\n').encode())
                connection.shutdown(socket.SHUT_WR)
                with connection.makefile('rb') as reader:
                    raw = reader.readline(65537)
                if len(raw) > 65536:
                    raise ValueError('oversized profile archive response')
                value = json.loads(raw)
                if 'error' in value:
                    raise ValueError(value['error'])
        else:
            # Trusted local acceptance uses the same verified action archive.
            from action_records import read_profile_archive
            value = read_profile_archive(context.session_root, **request)
        if (not isinstance(value, dict) or value.get('readonly') is not True
                or value.get('mode') != ref.mode or value.get('event_seq') != ref.event_seq
                or (file_ref is None and not isinstance(value.get('files'), list))
                or (file_ref is not None and value.get('file_ref') != file_ref)):
            raise ValueError('invalid profile archive response')
    except Exception as exc:
        raise ProtocolError('Saved public capture unavailable') from exc
    return {'status': 'available', 'profile_ref': profile_ref, **value}
