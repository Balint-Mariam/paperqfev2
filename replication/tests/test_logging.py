import io
import sys
import unittest
from unittest.mock import patch
from src.environment import Tee, end_log


class LoggingTests(unittest.TestCase):
    def test_shared_log_closes_once_and_both_terminals_remain_writable(self):
        output,error,log=io.StringIO(),io.StringIO(),io.StringIO()
        with patch.object(sys,'stdout',Tee(output,log)),patch.object(sys,'stderr',Tee(error,log)):
            sys.stdout.write('output');sys.stderr.write('error');end_log()
            self.assertTrue(log.closed)
            sys.stdout.write('after');sys.stderr.write('after')
            self.assertEqual(output.getvalue(),'outputafter');self.assertEqual(error.getvalue(),'errorafter')
