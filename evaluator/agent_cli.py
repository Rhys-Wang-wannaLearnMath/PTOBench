"""Five Agent operations; explicit Session environment required. No bootstrap."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import session
import session_actions


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise session.SessionError(message, 'invalid_request')


def _emit_result(result):
    payload = (json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False) + '\n').encode()
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        print(payload.decode(), end='', flush=True)
        return
    blocking = os.get_blocking(fd)
    try:
        os.set_blocking(fd, False)
        offset = 0
        deadline = time.monotonic() + 2
        while offset < len(payload) and time.monotonic() < deadline:
            try:
                offset += os.write(fd, payload[offset:])
            except BlockingIOError:
                time.sleep(.01)
            except BrokenPipeError:
                break
    finally:
        os.set_blocking(fd, blocking)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = Parser(prog='ptobench', allow_abbrev=False)
    commands = parser.add_subparsers(dest='action', required=True, parser_class=Parser)
    evaluate = commands.add_parser('evaluate', allow_abbrev=False)
    evaluate.add_argument('workspace', nargs='?', default='.')
    profile = commands.add_parser('profile', allow_abbrev=False)
    profile.add_argument('mode', choices=('runtime', 'pmu', 'compiler'))
    profile.add_argument('workspace', nargs='?', default='.')
    fork = commands.add_parser('fork', allow_abbrev=False)
    fork.add_argument('workspace', nargs='?', default='.')
    root = os.environ.get('PTOBENCH_SESSION_ROOT')
    try:
        args = parser.parse_args(argv)
        if not root:
            raise session.SessionError('PTOBENCH_SESSION_ROOT is required', 'invalid_request')
    except session.SessionError as error:
        if root:
            result = session_actions.reject_request(root, argv[0] if argv else 'invalid_request', error)
        else:
            result = session_actions._error_result(error)
    else:
        if args.action == 'evaluate':
            result = session_actions.run_evaluate(root, args.workspace)
        elif args.action == 'profile':
            result = session_actions.run_profile(root, args.workspace, args.mode)
        else:
            result = session_actions.fork_workspace(root, args.workspace)
    try:
        _emit_result(result)
    except (BrokenPipeError, BlockingIOError):
        pass
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
