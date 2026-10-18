#!/usr/bin/env python3
"""Local Pi/stdio-MCP transport for CANNBot's native stage sessions."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluator')]

from modes.cannbot.runtime import exclusive, load_config, run_dispatch, save


def specifications(allow_task, config):
    schema = lambda properties, required=(): {'type': 'object', 'properties': properties,
                                             'required': list(required), 'additionalProperties': True}
    tools = [
        {'name': 'skill', 'description': 'Read an unchanged upstream skill and its source path. Available: ' + ', '.join(config['skills']),
         'inputSchema': schema({'name': {'type': 'string'}}, ('name',))},
        {'name': 'todowrite', 'description': 'Persist your Todo list unchanged in the method work directory.',
         'inputSchema': schema({'todos': {'type': 'array', 'items': {}}}, ('todos',))},
        {'name': 'todoread', 'description': 'Read your persisted Todo list.', 'inputSchema': schema({})},
    ]
    if allow_task:
        tools.insert(0, {'name': 'Task', 'description': 'Dispatch an original pypto-op-optimizer or pypto-op-verifier role in a fresh native session. Pass all stage instructions and handoff information explicitly in prompt. Sessions share source and PTOBench budget; the original answer or error is returned. Tasks are serialized.',
            'inputSchema': schema({'subagent_type': {'type': 'string', 'enum': ['pypto-op-optimizer', 'pypto-op-verifier']},
                                   'prompt': {'type': 'string'}, 'description': {'type': 'string'},
                                   'workspace': {'type': 'string'}}, ('subagent_type', 'prompt'))})
    return tools


def tool_result(text, *, error=False, details=None):
    result = {'content': [{'type': 'text', 'text': text}], 'isError': error}
    if details is not None:
        result['details'] = details
    return result


def call(config, dispatch, name, arguments, call_id, allow_task, cancel=None, raw_request=None):
    directory = Path(config['run_root']) / 'method/tools' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    save(directory / 'input.json', {'dispatch_id': dispatch, 'tool_call_id': call_id,
                                   'call_id_source': 'pi_native' if config['profile']['backend'] == 'pi' else 'mcp_request',
                                   'name': name, 'arguments': arguments, 'request': raw_request})
    try:
        if not isinstance(arguments, dict):
            raise ValueError('Tool arguments must be an object')
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Tool call cancelled')
        if name == 'Task' and allow_task:
            role, prompt = arguments['subagent_type'], arguments['prompt']
            if role not in ('pypto-op-optimizer', 'pypto-op-verifier') or not isinstance(prompt, str):
                raise ValueError('Task requires an original stage role and a text prompt')
            child_config = dict(config)
            if 'workspace' in arguments:
                target = Path(arguments['workspace']).resolve()
                # This is only a public workspace binding check. The CLI remains
                # responsible for authoritative registration/Case enforcement.
                binding = json.loads((target / '.benchmark_context/workspace.json').read_text())
                current = json.loads((Path(config['workspace']) / '.benchmark_context/workspace.json').read_text())
                target.relative_to(Path(config['run_root']) / 'workspaces')
                if binding.get('case_id') != current.get('case_id') or binding.get('workspace') != str(target):
                    raise ValueError('Task workspace must belong to the same public Run')
                child_config['workspace'] = str(target)
            with exclusive(Path(config['work_dir']) / '.task.lock', cancel):
                result = run_dispatch(child_config, role, prompt, parent=dispatch,
                                      tool_call_id=call_id, cancel=cancel, task_input=arguments)
            text = result['answer']
            if result['is_error']:
                text = (text + '\n' if text else '') + json.dumps({key: value for key, value in result.items() if key != 'answer'}, ensure_ascii=False)
            value = tool_result(text, error=result['is_error'], details=result)
        elif name == 'skill':
            requested = arguments['name']
            try:
                source = Path(config['skills'][requested])
            except KeyError:
                # The role-prefixed spelling has appeared in native requests;
                # resolve it only against the frozen skill index.
                alias = requested.replace('pypto-op-orchestration-manual',
                                          'pypto-orchestration-manual', 1)
                source = Path(config['skills'][alias])
            value = tool_result('Source: ' + str(source) + '\n\n' + source.read_text(), details={'source': str(source)})
        elif name in ('todowrite', 'todoread'):
            path = Path(config['work_dir']) / 'todos.json'
            with exclusive(path.with_suffix('.lock'), cancel):
                if name == 'todowrite':
                    if not isinstance(arguments.get('todos'), list):
                        raise ValueError('todos must be an array')
                    save(path, arguments['todos'])
                todos = json.loads(path.read_text()) if path.exists() else []
            value = tool_result(json.dumps(todos, ensure_ascii=False), details={'path': str(path)})
        else:
            raise ValueError('Tool is unavailable in this native role: ' + name)
    except Exception as error:
        value = tool_result(json.dumps({'type': type(error).__name__, 'message': str(error)}, ensure_ascii=False), error=True)
    save(directory / 'result.json', value)
    return value


def mcp(config, dispatch, allow_task):
    """Newline JSON-RPC; keep reading so cancellation reaches a running Task."""
    lock, pending, workers = threading.Lock(), {}, []
    def send(value):
        with lock:
            sys.stdout.write(json.dumps(value, ensure_ascii=False) + '\n')
            sys.stdout.flush()
    def execute(request, cancel):
        try:
            params = request.get('params', {})
            value = call(config, dispatch, params['name'], params.get('arguments', {}),
                         str(request['id']), allow_task, cancel, request)
            # Pi's details are not an MCP CallToolResult field.
            send({'jsonrpc': '2.0', 'id': request['id'], 'result': {key: value[key] for key in ('content', 'isError')}})
        except Exception as error:
            send({'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32602, 'message': str(error)}})
        finally:
            with lock:
                pending.pop(request['id'], None)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except ValueError:
                send({'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Invalid JSON'}})
                continue
            method, identity = request.get('method'), request.get('id')
            if method == 'notifications/cancelled':
                save(Path(config['run_root']) / 'method/tools/cancellations' / (uuid.uuid4().hex + '.json'),
                     {'dispatch_id': dispatch, 'request': request})
                with lock:
                    event = pending.get(request.get('params', {}).get('requestId'))
                if event is not None:
                    event.set()
                continue
            if identity is None:
                continue
            if method == 'initialize':
                value = {'protocolVersion': request.get('params', {}).get('protocolVersion', '2024-11-05'),
                         'capabilities': {'tools': {}}, 'serverInfo': {'name': 'cannbot', 'version': '1.0.0'}}
            elif method == 'ping':
                value = {}
            elif method == 'tools/list':
                value = {'tools': specifications(allow_task, config)}
            elif method == 'tools/call':
                event = threading.Event()
                with lock:
                    pending[identity] = event
                worker = threading.Thread(target=execute, args=(request, event), daemon=True)
                workers.append(worker)
                worker.start()
                continue
            else:
                send({'jsonrpc': '2.0', 'id': identity, 'error': {'code': -32601, 'message': 'Method not found'}})
                continue
            send({'jsonrpc': '2.0', 'id': identity, 'result': value})
    finally:
        with lock:
            for event in pending.values():
                event.set()
        for worker in workers:
            worker.join(timeout=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('transport', choices=('call', 'mcp', 'list'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--dispatch', required=True)
    parser.add_argument('--allow-task', action='store_true')
    options = parser.parse_args()
    config = load_config(options.config)
    if options.transport == 'mcp':
        mcp(config, options.dispatch, options.allow_task)
    elif options.transport == 'list':
        print(json.dumps(specifications(options.allow_task, config), ensure_ascii=False))
    else:
        cancel = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, frame: cancel.set())
        request = json.load(sys.stdin)
        value = call(config, options.dispatch, request['name'], request.get('arguments', {}),
                     request['tool_call_id'], options.allow_task, cancel, request)
        print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
