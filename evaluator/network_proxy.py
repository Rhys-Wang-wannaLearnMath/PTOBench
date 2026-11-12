"""Agent network bridge: isolated Agents reach only their configured model API."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import signal
import ssl
from urllib.parse import urlsplit


async def pipe(source, destination):
    try:
        while data := await source.read(65536):
            destination.write(data)
            await destination.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        destination.close()


def target(value, *, connect):
    if connect:
        parsed = urlsplit('//' + value)
        scheme = 'https'
    else:
        parsed = urlsplit(value)
        scheme = parsed.scheme
    if scheme not in ('http', 'https') or not parsed.hostname:
        return None
    try:
        port = parsed.port or (443 if scheme == 'https' else 80)
    except ValueError:
        return None
    return parsed.hostname.lower().rstrip('.'), port


class ModelProxy:
    def __init__(self, allowed):
        self.allowed = allowed

    async def handle(self, client_reader, client_writer):
        try:
            first = await asyncio.wait_for(client_reader.readline(), 10)
            method, destination, version = first.decode('latin-1').strip().split(' ', 2)
            connect = method.upper() == 'CONNECT'
            endpoint = target(destination, connect=connect)
            headers = await asyncio.wait_for(client_reader.readuntil(b'\r\n\r\n'), 10)
            if endpoint != self.allowed:
                client_writer.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                await client_writer.drain()
                return
            if not connect:
                host_header = next((line.split(b':', 1)[1].strip().decode('latin-1')
                                    for line in headers.split(b'\r\n')
                                    if line.lower().startswith(b'host:')), '')
                if target(host_header, connect=True) != self.allowed:
                    client_writer.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                    await client_writer.drain()
                    return
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(*self.allowed), 30)
            if connect:
                client_writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                await client_writer.drain()
            else:
                url = urlsplit(destination)
                path = url.path or '/'
                if url.query:
                    path += '?' + url.query
                upstream_writer.write(f'{method} {path} {version}\r\n'.encode() + headers)
                await upstream_writer.drain()
            await asyncio.gather(pipe(client_reader, upstream_writer),
                                 pipe(upstream_reader, client_writer))
        except (OSError, asyncio.TimeoutError, ValueError, asyncio.IncompleteReadError):
            client_writer.write(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            try:
                await client_writer.drain()
            except OSError:
                pass
        finally:
            client_writer.close()


class ModelAPIGateway:
    """Forward only model inference requests and add the host-held credential."""
    MAX_HEADER = 64 * 1024
    MAX_BODY = 256 * 1024 * 1024

    def __init__(self, base_url, api, api_key):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('invalid model API base URL')
        if api not in ('openai-responses', 'openai-completions'):
            raise ValueError('unsupported model API protocol')
        self.scheme, self.host = parsed.scheme, parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        self.prefix = parsed.path.rstrip('/')
        suffix = '/responses' if api == 'openai-responses' else '/chat/completions'
        self.paths = {self.prefix + suffix, self.prefix + '/models'}
        self.api, self.api_key = api, api_key

    @staticmethod
    async def _headers(reader):
        block = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 15)
        if len(block) > ModelAPIGateway.MAX_HEADER:
            raise ValueError('HTTP header too large')
        lines = block[:-4].split(b'\r\n')
        headers = []
        for line in lines[1:]:
            name, separator, value = line.partition(b':')
            if not separator:
                raise ValueError('invalid HTTP header')
            headers.append((name.decode('latin-1').strip().lower(), line))
        return lines[0], headers

    @classmethod
    async def _request_body(cls, reader, headers):
        lengths = [line.split(b':', 1)[1].strip() for name, line in headers if name == 'content-length']
        transfers = [line.split(b':', 1)[1].strip().lower() for name, line in headers if name == 'transfer-encoding']
        if len(lengths) > 1 or (lengths and transfers):
            raise ValueError('ambiguous HTTP body length')
        if transfers:
            if transfers != [b'chunked']:
                raise ValueError('unsupported HTTP transfer encoding')
            chunks = bytearray()
            total = 0
            while True:
                line = await asyncio.wait_for(reader.readline(), 15)
                total += len(line)
                if total > cls.MAX_BODY:
                    raise ValueError('HTTP body too large')
                size = int(line.split(b';', 1)[0].strip(), 16)
                chunks.extend(line)
                if size == 0:
                    while True:
                        trailer = await asyncio.wait_for(reader.readline(), 15)
                        chunks.extend(trailer)
                        total += len(trailer)
                        if total > cls.MAX_BODY:
                            raise ValueError('HTTP body too large')
                        if trailer == b'\r\n':
                            return bytes(chunks)
                if total + size + 2 > cls.MAX_BODY:
                    raise ValueError('HTTP body too large')
                chunk = await asyncio.wait_for(reader.readexactly(size + 2), 60)
                chunks.extend(chunk)
                total += len(chunk)
        if lengths:
            size = int(lengths[0])
            if size < 0 or size > cls.MAX_BODY:
                raise ValueError('HTTP body too large')
            return await asyncio.wait_for(reader.readexactly(size), 120)
        return b''

    async def handle(self, client_reader, client_writer):
        upstream_writer = None
        try:
            line, headers = await self._headers(client_reader)
            method, path, version = line.decode('latin-1').split(' ', 2)
            request_path = urlsplit(path).path
            allowed = ((method == 'POST' and request_path in self.paths)
                       or (method == 'GET' and request_path == self.prefix + '/models'))
            if not allowed or not path.startswith('/'):
                client_writer.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                await client_writer.drain()
                return

            body = await self._request_body(client_reader, headers)
            kept = [line for name, line in headers
                    if name not in ('host', 'authorization', 'x-api-key', 'api-key',
                                    'proxy-authorization', 'connection', 'proxy-connection')]
            kept.extend((f'Host: {self.host}:{self.port}'.encode('latin-1'),
                         f'Authorization: Bearer {self.api_key}'.encode('latin-1'),
                         b'Connection: close'))
            context = ssl.create_default_context() if self.scheme == 'https' else None
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port, ssl=context,
                                        server_hostname=self.host if context else None), 30)
            upstream_writer.write((f'{method} {path} {version}\r\n'.encode('latin-1')
                                   + b'\r\n'.join(kept) + b'\r\n\r\n' + body))
            await upstream_writer.drain()
            response_line, response_headers = await self._headers(upstream_reader)
            client_writer.write(response_line + b'\r\n'
                                + b'\r\n'.join(line for _, line in response_headers)
                                + b'\r\n\r\n')
            await client_writer.drain()
            await pipe(upstream_reader, client_writer)
        except (OSError, asyncio.TimeoutError, ValueError, asyncio.IncompleteReadError,
                asyncio.LimitOverrunError, UnicodeError):
            if not client_writer.is_closing():
                client_writer.write(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                try:
                    await client_writer.drain()
                except OSError:
                    pass
        finally:
            if upstream_writer is not None:
                upstream_writer.close()
            client_writer.close()


async def serve(socket_path, host, port):
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    server = await asyncio.start_unix_server(ModelProxy((host, port)).handle, path=path)
    import isolation
    isolation._agent_shared_socket(path)
    async with server:
        await server.serve_forever()


async def serve_api(socket_path, base_url, api):
    api_key = os.environ.get('PTOBENCH_UPSTREAM_API_KEY')
    if not api_key:
        raise RuntimeError('model API credential is unavailable')
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    server = await asyncio.start_unix_server(
        ModelAPIGateway(base_url, api, api_key).handle, path=path)
    import isolation
    isolation._agent_shared_socket(path)
    async with server:
        await server.serve_forever()


async def bridge(socket_path, port, api_socket, api_port, command, delete_runtime_files):
    servers = []

    def relay_to(socket_path):
        async def relay(reader, writer):
            try:
                upstream_reader, upstream_writer = await asyncio.open_unix_connection(socket_path)
            except OSError:
                writer.close()
                return
            await asyncio.gather(pipe(reader, upstream_writer), pipe(upstream_reader, writer))
        return relay

    if socket_path:
        servers.append(await asyncio.start_server(relay_to(socket_path), '127.0.0.1', port))
    if api_socket:
        servers.append(await asyncio.start_server(relay_to(api_socket), '127.0.0.1', api_port))
    if not servers:
        raise ValueError('at least one network bridge socket is required')
    if delete_runtime_files:
        Path(__file__).unlink(missing_ok=True)
    process = await asyncio.create_subprocess_exec(*command)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, process.send_signal, sig)
    try:
        return await process.wait()
    finally:
        for server in servers:
            server.close()
            await server.wait_closed()
        if process.returncode is None:
            process.terminate()
            await process.wait()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    host_proxy = commands.add_parser('serve')
    host_proxy.add_argument('--socket', required=True)
    host_proxy.add_argument('--host', required=True)
    host_proxy.add_argument('--port', type=int, required=True)
    api_proxy = commands.add_parser('serve-api')
    api_proxy.add_argument('--socket', required=True)
    api_proxy.add_argument('--base-url', required=True)
    api_proxy.add_argument('--api', required=True)
    agent = commands.add_parser('agent')
    agent.add_argument('--socket')
    agent.add_argument('--port', type=int, default=18080)
    agent.add_argument('--api-socket')
    agent.add_argument('--api-port', type=int, default=18081)
    agent.add_argument('--delete-runtime-files', action='store_true')
    agent.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'serve':
        asyncio.run(serve(args.socket, args.host, args.port))
        return 0
    if args.command == 'serve-api':
        asyncio.run(serve_api(args.socket, args.base_url, args.api))
        return 0
    command = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not command:
        parser.error('agent requires a command after --')
    return asyncio.run(bridge(args.socket, args.port, args.api_socket, args.api_port,
                              command, args.delete_runtime_files))


if __name__ == '__main__':
    raise SystemExit(main())
