"""Repository-local experiment CLI: run and read-only status."""
import argparse
import sys
import session
from agent_cli import _emit_result
from launcher.methods import MethodError
from launcher.config import ConfigError
from launcher.runner import run, status


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise session.SessionError(message, 'invalid_request')


class Payload(dict):
    def to_dict(self):
        return self


def main(argv=None):
    parser = Parser(prog='pto-launcher', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    commands.add_parser('run', allow_abbrev=False).add_argument('experiment')
    commands.add_parser('status', allow_abbrev=False).add_argument('session_root')
    try:
        args = parser.parse_args(argv)
        if args.command == 'run':
            result, code = run(args.experiment)
        else:
            result, code = status(args.session_root), 0
    except Exception as error:
        error_code = getattr(error, 'error_code', 'infrastructure_error')
        # Avoid reflecting untrusted YAML or host authentication output.
        message = (str(error) if isinstance(error, (ConfigError, MethodError))
                   else 'Launcher request failed; check configuration and backend readiness')
        result = {'status': 'failed', 'error_code': error_code, 'message': message}
        code = 2 if error_code == 'invalid_request' else 5
    _emit_result(Payload(result))
    return code


if __name__ == '__main__':
    sys.exit(main())
