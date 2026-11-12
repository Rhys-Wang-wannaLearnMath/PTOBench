import asyncio
import grp
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import network_proxy
from network_proxy import ModelAPIGateway


class SharedSocketTests(unittest.TestCase):
    def test_model_gateway_socket_is_shared_but_parent_is_not_listable(self):
        async def check():
            with tempfile.TemporaryDirectory(prefix='model-gateway-socket-') as temporary:
                path = Path(temporary) / 'api.sock'
                with patch.dict(os.environ, {'PTOBENCH_UPSTREAM_API_KEY': 'fixture-secret'}):
                    server = asyncio.create_task(network_proxy.serve_api(
                        path, 'https://example.invalid/v1', 'openai-responses'))
                    try:
                        for _ in range(100):
                            if path.exists():
                                break
                            await asyncio.sleep(.01)
                        self.assertTrue(path.exists())
                        endpoint = path.stat()
                        directory = path.parent.stat()
                        self.assertEqual(endpoint.st_uid, os.getuid())
                        self.assertEqual(endpoint.st_gid, grp.getgrnam('ptobench-share').gr_gid)
                        self.assertEqual(stat.S_IMODE(endpoint.st_mode), 0o660)
                        self.assertEqual(directory.st_gid, grp.getgrnam('ptobench-share').gr_gid)
                        self.assertEqual(stat.S_IMODE(directory.st_mode), 0o710)
                    finally:
                        server.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await server

        asyncio.run(check())


class ChunkedRequestLimitTests(unittest.TestCase):
    def test_rejects_oversized_chunk_before_reading_its_payload(self):
        class Reader:
            async def readline(self):
                return b'9\r\n'

            async def readexactly(self, _size):
                raise AssertionError('oversized chunk payload must not be read')

        headers = [('transfer-encoding', b'Transfer-Encoding: chunked')]
        with patch.object(ModelAPIGateway, 'MAX_BODY', 8):
            with self.assertRaisesRegex(ValueError, 'HTTP body too large'):
                asyncio.run(ModelAPIGateway._request_body(Reader(), headers))

    def test_counts_chunk_trailers_toward_body_limit(self):
        class Reader:
            lines = iter((b'0\r\n', b'X: Y\r\n'))

            async def readline(self):
                return next(self.lines)

        headers = [('transfer-encoding', b'Transfer-Encoding: chunked')]
        with patch.object(ModelAPIGateway, 'MAX_BODY', 8):
            with self.assertRaisesRegex(ValueError, 'HTTP body too large'):
                asyncio.run(ModelAPIGateway._request_body(Reader(), headers))


if __name__ == '__main__':
    unittest.main()
