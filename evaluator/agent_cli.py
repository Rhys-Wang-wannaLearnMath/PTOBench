"""Five Agent operations; explicit Session environment required. No bootstrap."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import profile_modes
import session
import session_actions


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise session.SessionError(message, 'invalid_request')


def _emit_result(result):
    payload = (json.dumps(result.to_agent_dict(), ensure_ascii=False, allow_nan=False) + '\n').encode()
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
    root = os.environ.get('PTOBENCH_SESSION_ROOT')
    try:
        session_record = session.read_session(root) if root else None
        enabled_profiles = (session_record['profiling_modes'] if session_record
                            else list(profile_modes.public_enabled_modes()))
        profile_strategy = (session_record.get('profile_strategy', profile_modes.DEFAULT_PROFILE_STRATEGY)
                            if session_record else profile_modes.DEFAULT_PROFILE_STRATEGY)
    except Exception as error:
        _emit_result(session_actions._error_result(error))
        return session_actions._error_result(error).exit_code
    parser = Parser(prog='ptobench', allow_abbrev=False)
    commands = parser.add_subparsers(dest='action', required=True, parser_class=Parser)
    evaluate = commands.add_parser('evaluate', allow_abbrev=False,
                                  help='Check correctness and measure Candidate latency.',
                                  description=('Check correctness and measure Candidate latency; consumes one hardware action '
                                               'per accepted action, including failed or interrupted actions.'))
    evaluate.add_argument('workspace', nargs='?', default='.', help='Registered workspace root (default: current directory).')
    if enabled_profiles:
        profile = commands.add_parser('profile', allow_abbrev=False,
                                      help=('Collect combined runtime, PMU and compiler diagnostics.'
                                            if profile_strategy == 'combined'
                                            else 'Collect runtime, PMU or compiler diagnostics.'),
                                      description=(('Collect combined runtime, PMU and compiler diagnostics. '
                                                   if profile_strategy == 'combined' else 'Collect diagnostics. ')
                                                   + 'Profile does not check output correctness; '
                                                   'use `ptobench evaluate` to check correctness and clean latency. '
                                                   'Each accepted Profile consumes one hardware action; '
                                                   'all further Evaluate, Profile and Fork requests are rejected when the hardware-action budget is exhausted.'))
        if profile_strategy == 'combined':
            profile.add_argument('workspace', nargs='?', default='.', help='Registered workspace root (default: current directory).')
        else:
            profile.add_argument('mode', choices=enabled_profiles,
                                 help='runtime: scheduling/dependencies; pmu: hardware counters; compiler: compiler/IR diagnostics.')
            profile.add_argument('workspace', nargs='?', default='.', help='Registered workspace root (default: current directory).')
    fork = commands.add_parser('fork', allow_abbrev=False, help='Copy source into a new registered workspace.',
                               description='Copy source into a new registered workspace and return its path; consumes no hardware-action budget and is recorded as a Session operation. Available only before the hardware-action budget is exhausted.')
    fork.add_argument('workspace', nargs='?', default='.', help='Registered parent workspace root (default: current directory).')
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
            result = session_actions.run_profile(
                root, args.workspace,
                'combined' if profile_strategy == 'combined' else args.mode,
            )
        else:
            result = session_actions.fork_workspace(root, args.workspace)
    try:
        _emit_result(result)
    except (BrokenPipeError, BlockingIOError):
        pass
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
