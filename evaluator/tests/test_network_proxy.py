import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from network_proxy import ModelAPIGateway


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
