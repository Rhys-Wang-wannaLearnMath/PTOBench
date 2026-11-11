#!/usr/bin/python3
"""Small untrusted-side CLI transport; it has no host execution authority."""
import json
import os
import socket
import sys


def main():
    request = {'argv': sys.argv[1:], 'cwd': os.getcwd()}
    for name in ('PTOBENCH_AGENT_DISPATCH_ID', 'PTOBENCH_AGENT_PARENT_TOOL_CALL_ID'):
        value = os.environ.get(name)
        if value:
            request[name] = value
    request = json.dumps(request).encode() + b'\n'
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(os.environ['PTOBENCH_RPC_SOCKET'])
        connection.sendall(request)
        stream = connection.makefile('rb')
        # Length-prefixed response lines preserve arbitrary evaluator output.
        code = 5
        for line in stream:
            event = json.loads(line)
            if 'output' in event:
                print(event['output'], end='', flush=True)
            if 'exit_code' in event:
                return event['exit_code']
        return code


if __name__ == '__main__':
    raise SystemExit(main())
