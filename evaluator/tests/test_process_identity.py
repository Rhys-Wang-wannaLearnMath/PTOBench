"""A /proc entry may disappear after opening it but before reading stat."""
import errno
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import integrity
import session


class ProcessIdentityRaceTests(unittest.TestCase):
    def test_disappearing_process_is_absent_for_both_proc_errors(self):
        for error in (FileNotFoundError(errno.ENOENT, 'removed before open'),
                      ProcessLookupError(errno.ESRCH, 'exited during stat read')):
            with self.subTest(error=type(error).__name__), mock.patch.object(Path, 'read_text', side_effect=error):
                self.assertIsNone(session.process_identity(12345))

    def test_unreadable_identity_still_blocks_cleanup(self):
        with mock.patch.object(Path, 'read_text', side_effect=PermissionError(errno.EACCES, 'cannot inspect')):
            with self.assertRaises(integrity.InfrastructureError):
                session.process_identity(12345)


if __name__ == '__main__':
    unittest.main()
