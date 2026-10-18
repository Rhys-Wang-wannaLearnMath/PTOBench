"""Fresh native sessions; no phase routing or optimization policy lives here."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import uuid

from modes._shared.config import render
from modes._shared import recording, knowledge
from modes.cannbot import resources
from workspace import source_manifest


def load_config(path):
    value = json.loads(Path(path).read_text())
    value['config_path'] = str(Path(path).absolute())
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def exclusive(path, cancel=None):
    """Serialize Tasks sharing mutable source, while allowing cancellation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as lock:
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Task cancelled before dispatch')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancel is not None:
                    cancel.wait(.05)
                else:
                    time.sleep(.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def source_state(workspace):
    try:
        return source_manifest(Path(workspace))
    except Exception as error:
        return {'error': str(error)}


def command(config, directory, role, task, allow_task):
    original = Path(config['upstream']) / config['roles'][role]
    role_text = original.read_text()
    if role == 'pypto-op-optimizer':
        # Upstream marks its main performance skill as automatically loaded.
        role_text += '\n\n' + Path(config['skills']['pypto-op-perf-tune']).read_text()
    system = directory / 'system.md'
    system.write_text(role_text)
    # Pi has a native system-prompt file option. Codex receives the exact role
    # bytes as part of its initial message, with the Task input kept separately.
    if not allow_task:
        # Every fresh stage gets public environment information independently
        # of the parent's Task handoff. Original role/skill bytes stay intact.
        task = ("Read .benchmark_context/README.md for the Case contract, Bench CLI and result locations.\n"
                "Compiler tuning is editable in workspace-root tuning.json; its defaults are PYPTO + false.\n"
                "PATH and the Bench Session environment are already set.\n"
                + knowledge.notice(config['run_root'])
                + f"Share this Bench Session's {config['hardware_budget']}-action exploration budget; "
                  "this Agent stage does not receive a new budget.\n\n" + task)
    prompt = task if config['profile']['backend'] == 'pi' else role_text + '\n\n' + task
    argv, configured_env = render(config['profile'], config['workspace'], directory, prompt)
    environment = {**os.environ, **configured_env,
                   'PTOBENCH_SESSION_ROOT': config['run_root'],
                   'CANNBOT_CONFIG': config['config_path'],
                   'CANNBOT_DISPATCH': directory.name,
                   'CANNBOT_ALLOW_TASK': '1' if allow_task else '0',
                   'CANNBOT_PYTHON': config['python'],
                   'CANNBOT_TOOLS': str(resources.DIRECTORY / 'tools.py')}
    additions = []
    if config['profile']['backend'] == 'pi':
        additions = ['--extension', str(resources.DIRECTORY / 'bridge/pi_task.ts'),
                     '--append-system-prompt', str(system)]
    else:
        args = [str(resources.DIRECTORY / 'tools.py'), 'mcp', '--config', config['config_path'],
                '--dispatch', directory.name]
        if allow_task:
            args.append('--allow-task')
        overrides = {
            'mcp_servers.cannbot.command': config['python'],
            'mcp_servers.cannbot.args': args,
            'mcp_servers.cannbot.tool_timeout_sec': 43200,
        }
        for key, value in overrides.items():
            additions += ['-c', key + '=' + json.dumps(value, ensure_ascii=False)]
    # Place options before the positional prompt, including custom profiles.
    index = config['profile']['argv'].index('{prompt}')
    argv[index:index] = additions
    return argv, environment, prompt, original


def original_answer(path, backend):
    answer, error = None, None
    for event in recording._json_lines(path):
        if backend == 'pi' and event.get('type') == 'message_end':
            message = event.get('message', {})
            if message.get('role') == 'assistant':
                parts = message.get('content', [])
                answer = ''.join(part.get('text', '') for part in parts if isinstance(part, dict) and part.get('type') == 'text') if isinstance(parts, list) else str(parts)
                error = (message.get('errorMessage') or message['stopReason']) if message.get('stopReason') in ('error', 'aborted') else None
        if backend == 'codex':
            item = event.get('item', {})
            if event.get('type') == 'item.completed' and item.get('type') == 'agent_message':
                answer = item.get('text', '')
            if event.get('type') in ('turn.failed', 'error'):
                error = event.get('error', event.get('message', event))
            if event.get('type') == 'turn.completed':
                error = None
    return answer, error


def descendants(pid):
    """Pin only descendants of this native process for per-Task cancellation."""
    from session import process_identity
    children = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            children.setdefault(int(fields[1]), []).append(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    pending, found = [pid], []
    while pending:
        current = pending.pop()
        identity = process_identity(current)
        if identity:
            found.append(identity)
            pending.extend(children.get(current, []))
    return found


def stop_tree(identities, sig):
    from session import process_identity
    for identity in reversed(identities):
        if process_identity(identity['pid']) == identity:
            try:
                os.kill(identity['pid'], sig)
            except ProcessLookupError:
                pass


def run_dispatch(config, role, task, *, parent, tool_call_id, forward=False, cancel=None, task_input=None):
    if role not in config['roles']:
        raise ValueError('Unknown upstream role: ' + str(role))
    directory = Path(config['run_root']) / 'method/dispatches' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    _, configured_env = render(config['profile'], config['workspace'], directory, '')
    native_profile = {**config['profile'], 'env': configured_env}
    state = {'schema_version': 1, 'id': directory.name, 'parent_dispatch_id': parent,
             'tool_call_id': tool_call_id, 'role': role,
             'tool_call_id_source': 'pi_native' if config['profile']['backend'] == 'pi' else 'mcp_request' if parent is not None else None,
             'workspace': config['workspace'], 'session_root': config['run_root'],
             'profile': config['profile'], 'native_profile': native_profile,
             'task_input': task_input, 'started_at_unix': time.time(),
             'started': False, 'status': 'starting'}
    # Persist dispatch intent before reading large sources or role resources.
    save(directory / 'dispatch.json', state)
    cancelled = False
    exit_code, answer, error = None, None, None
    process = None
    try:
        save(directory / 'config.json', {key: value for key, value in config.items() if key != 'config_path'})
        config = load_config(directory / 'config.json')
        (directory / 'task.txt').write_text(task)
        supplied = {key: value for key, value in (task_input or {}).items() if key not in ('prompt', 'subagent_type')}
        native_task = task + ('\n\nTask arguments:\n' + json.dumps(supplied, ensure_ascii=False) if supplied else '')
        argv, environment, prompt, original = command(config, directory, role, native_task, parent is None)
        (directory / 'prompt.txt').write_text(prompt)
        state.update(argv=argv, role_path=str(original),
                     role_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                     system_path=str(directory / 'system.md'),
                     system_sha256=hashlib.sha256((directory / 'system.md').read_bytes()).hexdigest(),
                     source_before=source_state(config['workspace']))
        save(directory / 'dispatch.json', state)
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Task cancelled before native process start')
        # Inherit Launcher's process group: no detached child or fresh budget.
        process = subprocess.Popen(argv, cwd=config['workspace'], env=environment, shell=False,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        from session import process_identity
        state.update(started=True, status='running', process=process_identity(process.pid))
        save(directory / 'dispatch.json', state)
        with selectors.DefaultSelector() as selector, open(directory / 'stdout.log', 'wb', buffering=0) as out, open(directory / 'stderr.log', 'wb', buffering=0) as err:
            for pipe, stream, destination in ((process.stdout, out, 1), (process.stderr, err, 2)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, (stream, destination))
            ended_at, stop_at, owned = None, None, []
            while selector.get_map() or process.poll() is None:
                if cancel is not None and cancel.is_set() and stop_at is None:
                    cancelled = True
                    owned = descendants(process.pid)
                    stop_tree(owned, signal.SIGTERM)
                    stop_at = time.monotonic()
                if stop_at is not None and time.monotonic() - stop_at > 1:
                    stop_tree(owned, signal.SIGKILL)
                if process.poll() is not None:
                    ended_at = ended_at or time.monotonic()
                    if time.monotonic() - ended_at > 1:
                        break  # inherited pipe FDs must not stall finalization
                for key, _ in selector.select(.05):
                    try:
                        data = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if data:
                        key.data[0].write(data)
                        if forward:
                            os.write(key.data[1], data)
                    else:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
            for key in list(selector.get_map().values()):
                key.fileobj.close()
        exit_code = process.wait()
        answer, error = original_answer(directory / 'stdout.log', config['profile']['backend'])
    except Exception as failure:
        if process is not None and process.poll() is None:
            owned = descendants(process.pid)
            stop_tree(owned, signal.SIGKILL)
            process.wait()
        error = {'type': type(failure).__name__, 'message': str(failure)}
        cancelled = isinstance(failure, InterruptedError)
    result = {'dispatch_id': directory.name, 'exit_code': exit_code, 'answer': answer,
              'error': error, 'cancelled': cancelled}
    result['is_error'] = cancelled or error is not None or exit_code != 0 or answer is None
    if answer is None and error is None:
        result['error'] = 'Native session produced no final assistant answer; see original logs.'
    save(directory / 'result.json', result)
    state.update(status='cancelled' if cancelled else 'failed' if result['is_error'] else 'completed',
                 ended_at_unix=time.time(), source_after=source_state(config['workspace']))
    save(directory / 'dispatch.json', state)
    # Capture before later sessions can rotate native histories. Final collect
    # also recovers interrupted dispatches which never reached this point.
    try:
        native = recording.collect(directory, native_profile, Path(config['workspace']),
                                   started=state['started'], close_reason=state['status'])
        save(directory / 'recording.json', native)
    except Exception as failure:
        save(directory / 'recording.json', {'status': 'missing', 'error': str(failure)})
    return result


def collect_dispatches(root, profile, started, close_reason):
    rows = []
    for directory in sorted((root / 'method/dispatches').glob('*')):
        if not directory.is_dir():
            continue
        if not (directory / 'dispatch.json').exists():
            rows.append({'dispatch_id': directory.name, 'parent_dispatch_id': None,
                         'tool_call_id': None, 'role': None, 'status': 'incomplete',
                         'path': str(directory.relative_to(root)),
                         'recording': {'status': 'missing', 'native_session_ids': [], 'files': [],
                                       'message': 'Dispatch metadata was not committed; original files retained.'}})
            continue
        state = json.loads((directory / 'dispatch.json').read_text())
        if state['status'] in ('starting', 'running'):
            state.update(status='interrupted', close_reason=close_reason,
                         collected_at_unix=time.time(), source_after=source_state(state['workspace']))
            save(directory / 'dispatch.json', state)
            answer, error = original_answer(directory / 'stdout.log', profile['backend']) if (directory / 'stdout.log').exists() else (None, None)
            save(directory / 'result.json', {'dispatch_id': directory.name, 'exit_code': None,
                 'answer': answer, 'error': error, 'is_error': True, 'interrupted': True,
                 'close_reason': close_reason})
        native = recording.collect(directory, state['native_profile'], Path(state['workspace']),
                                   started=state['started'], close_reason=state['status'])
        save(directory / 'recording.json', native)
        rows.append({'dispatch_id': directory.name, 'parent_dispatch_id': state['parent_dispatch_id'],
                     'tool_call_id': state['tool_call_id'], 'role': state['role'],
                     'status': state['status'], 'path': str(directory.relative_to(root)),
                     'recording': native})
    for directory in (root / 'method/tools').glob('*'):
        if directory.is_dir() and (directory / 'input.json').exists() and not (directory / 'result.json').exists():
            save(directory / 'interrupted.json', {'close_reason': close_reason, 'result_received': False})
    ids = [identity for row in rows for identity in row['recording']['native_session_ids']]
    status = 'recorded' if rows and all(row['recording']['status'] == 'recorded' for row in rows) else 'missing' if started else 'not_started'
    return {'schema_version': 1, 'run_id': root.name, 'backend': profile['backend'],
            'status': status, 'close_reason': close_reason, 'native_session_ids': ids,
            'dispatches': rows}
