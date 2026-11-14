"""Bounded local transport, serial read-only analysis and durable call auditing."""
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import socket
import socketserver
import threading
import uuid

REQUEST_LIMIT = 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def reply(value=None, error=None):
    return {'content': [{'type': 'text', 'text': error if error is not None else json.dumps(value, ensure_ascii=False, allow_nan=False)}],
            'isError': error is not None, **({'details': value} if error is None else {})}


def interrupt_unfinished(log_root):
    for request in Path(log_root).glob('*/request.json'):
        result = request.with_name('result.json')
        if not result.exists():
            record = json.loads(request.read_text())
            save(result, {**record, 'status': 'interrupted', 'ended_at': now(), 'profile': None,
                          'error': {'type': 'InterruptedError', 'message': 'Broker ended before a terminal result was saved'}})


class ToolServer:
    def __init__(self, adapter, socket_path: Path, log_root: Path):
        self.adapter = adapter
        self.path = Path(socket_path)
        self.log_root = Path(log_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self._guard = threading.Lock()
        self._serial = threading.Lock()
        self._states = {}
        self._precancelled = set()
        self._closing = threading.Event()
        self._close_lock = threading.Lock()
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(1)
                try:
                    line = self.rfile.readline(REQUEST_LIMIT + 1)
                    if len(line) > REQUEST_LIMIT or not line.endswith(b'\n'):
                        raise ValueError('request exceeds 1 MiB or is incomplete')
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError('request must be an object')
                    if request.get('action') == 'call':
                        result = owner._call(request, self.connection)
                    elif request.get('action') == 'cancel':
                        result = owner._cancel(request)
                    elif request.get('action') == 'registered':
                        result = owner._registered(request)
                    else:
                        raise ValueError('unknown transport action')
                except Exception as error:
                    result = reply(error=f'{type(error).__name__}: {error}')
                try:
                    self.wfile.write(json.dumps(result, ensure_ascii=False, allow_nan=False).encode() + b'\n')
                except OSError:
                    pass

        self._server = socketserver.ThreadingUnixStreamServer(str(self.path), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        self._thread.start()

    def _registered(self, request):
        expected = {s['name'] for s in self.adapter.specifications()}
        names, active = request.get('names', []), request.get('active_names', [])
        missing = sorted(expected - set(names))
        inactive = sorted(expected - set(active))
        record = {**request, 'recorded_at': now(), 'missing': missing, 'inactive': inactive,
                  'complete': not missing and not inactive}
        save(self.log_root.parent / 'analyzer_registration.json', record)
        if not record['complete']:
            return reply(error=f'Analyzer integration error: missing={missing}, inactive={inactive}')
        return reply(record)

    def _cancel(self, request):
        identity = request.get('tool_call_id')
        if not isinstance(identity, str):
            raise ValueError('tool_call_id must be a string')
        with self._guard:
            state = self._states.get(identity)
            if state is None:
                self._precancelled.add(identity)
            elif not state['done'].is_set():
                state['cancel'].set()
                save(state['directory'] / 'cancel.json', {**request, 'requested_at': now()})
        return reply({'cancel_requested': identity})

    def _call(self, request, connection):
        directory = self.log_root / uuid.uuid4().hex
        record = {**request, 'broker_id': directory.name, 'started_at': now()}
        save(directory / 'request.json', record)
        identity = request.get('tool_call_id')
        state = {'directory': directory, 'cancel': threading.Event(), 'done': threading.Event(),
                 'disconnected': threading.Event()}
        with self._guard:
            duplicate = isinstance(identity, str) and identity in self._states
            if isinstance(identity, str) and not duplicate:
                self._states[identity] = state
                if identity in self._precancelled:
                    self._precancelled.remove(identity)
                    state['cancel'].set()
            if request.get('cancelled') is True:
                state['cancel'].set()
        if duplicate:
            message = 'DuplicateCallError: tool_call_id was already received; not executed again'
            save(directory / 'result.json', {**record, 'status': 'error', 'ended_at': now(), 'profile': None,
                                            'error': {'type': 'DuplicateCallError', 'message': message}})
            return reply(error=message)

        def disconnected():
            try:
                if select.select([connection], [], [], 0)[0] and connection.recv(1, socket.MSG_PEEK) == b'':
                    state['disconnected'].set()
            except (OSError, ValueError):
                state['disconnected'].set()

        def monitor():
            while not state['done'].wait(.01):
                disconnected()
                if state['disconnected'].is_set():
                    break

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        output, error, profile = None, None, None
        try:
            with self._serial:
                if not self._closing.is_set() and not state['cancel'].is_set() and not state['disconnected'].is_set():
                    if not isinstance(identity, str) or not identity:
                        raise ValueError('tool_call_id must be a nonempty string')
                    output = self.adapter.call(request.get('name'), request.get('arguments'))
                    profile = output.get('profile') if isinstance(output, dict) else None
                    # JSON transport failures are typed call errors, with the selected Profile retained.
                    json.dumps(output, ensure_ascii=False, allow_nan=False)
        except Exception as caught:
            error = {'type': type(caught).__name__, 'message': str(caught)}
            profile = getattr(caught, 'profile', profile)
        finally:
            disconnected()
            with self._guard:
                if self._closing.is_set() or state['disconnected'].is_set():
                    status = 'interrupted'
                elif state['cancel'].is_set():
                    status = 'cancelled'
                else:
                    status = 'error' if error else 'success'
                result = {**record, 'status': status, 'ended_at': now(), 'profile': profile}
                if status == 'success':
                    result['output'] = output
                elif status in ('cancelled', 'interrupted'):
                    result['error'] = {'type': 'CancelledError' if status == 'cancelled' else 'InterruptedError',
                                       'message': 'Synchronous analysis, if started, finished; response discarded'}
                else:
                    result['error'] = error
                try:
                    save(directory / 'result.json', result)
                finally:
                    state['done'].set()
                    watcher.join()
        return reply(output) if status == 'success' else reply(error=f"{result['error']['type']}: {result['error']['message']}")

    def close(self) -> None:
        with self._close_lock:
            if self._closing.is_set():
                return
            self._closing.set()
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self.path.unlink(missing_ok=True)
