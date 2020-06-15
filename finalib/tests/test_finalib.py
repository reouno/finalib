import pandas as pd

from finalib import finalib as fl


def test_PurgedKFold_no_purge_no_embargo():
    df = pd.DataFrame({'A': [0, 1]})
    pkf = fl.PurgedKFold(n_splits=2)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([1], [0]), ([0], [1])]
