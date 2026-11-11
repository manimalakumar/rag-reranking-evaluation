"""Hermetic tutorial checks: no credentials, network, datasets or notebook execution."""
import os,socket,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'tests')]

def main():
    os.environ.update(HAYSTACK_TELEMETRY_ENABLED='False',RAGAS_DO_NOT_TRACK='true')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_input_preparation','test_live_tutorial','test_tutorial_views','test_notebook_display','test_notebooks','test_budget'])
    with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'MODEL_BUDGET_PATH':str(Path(tmp)/'budget.sqlite')}),patch.object(socket.socket,'connect',side_effect=AssertionError('Offline tests forbid network')):
        class Isolated(unittest.TextTestResult):
            def startTest(self,test):
                os.environ['MODEL_BUDGET_PATH']=str(Path(tmp)/(test.id()+'.sqlite'));super().startTest(test)
        result=unittest.TextTestRunner(verbosity=2,resultclass=Isolated).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun else 1
if __name__=='__main__':raise SystemExit(main())
