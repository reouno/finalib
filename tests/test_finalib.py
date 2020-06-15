import pandas as pd

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# pylint: disable=import-error
import finalib.finalib as fl

def test_PurgedKFold_no_purge_no_embargo():
    df = pd.DataFrame({'A': [0, 1]})
    pkf = fl.PurgedKFold(n_splits=2)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([1], [0]), ([0], [1])]
