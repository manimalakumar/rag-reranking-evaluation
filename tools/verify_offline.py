"""Run all migration tests with outbound sockets blocked and credentials removed."""
import os
from pathlib import Path
import socket
import sys
import unittest
import tempfile
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    os.chdir(PROJECT)
    os.environ['HAYSTACK_TELEMETRY_ENABLED'] = 'False'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
    os.environ.pop('OPENROUTER_API_KEY', None)
    os.environ.pop('ALLOW_MODEL_CALLS', None)
    os.environ.pop('NOTEBOOK_RUN_JUDGE', None)
    os.environ.pop('NOTEBOOK_ALLOW_MODEL_CALLS', None)
    os.environ.pop('NOTEBOOK_TESTS_PASSED', None)
    suite = unittest.defaultTestLoader.discover(str(PROJECT / 'tests'))
    with tempfile.TemporaryDirectory() as budget_directory, \
         patch.dict(os.environ, {'MODEL_BUDGET_PATH': str(Path(budget_directory) / 'budget.sqlite')}), \
         patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline tests forbid network')), \
         patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('Offline tests forbid network')):
        class IsolatedBudgetResult(unittest.TextTestResult):
            def startTest(self, test):
                os.environ['MODEL_BUDGET_PATH'] = str(Path(budget_directory) / (test.id() + '.sqlite'))
                super().startTest(test)
        result = unittest.TextTestRunner(verbosity=2, resultclass=IsolatedBudgetResult).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == '__main__':
    raise SystemExit(main())
