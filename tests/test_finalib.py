import pandas as pd

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# pylint: disable=import-error
import finalib as fl

def test_PurgedKFold_2_no_purge_no_embargo():
    df = pd.DataFrame({'A': [0, 1]})
    pkf = fl.PurgedKFold(n_splits=2)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([1], [0]), ([0], [1])]

def test_PurgedKFold_3_no_purge_no_embargo():
    df = pd.DataFrame({'A': [0, 1, 2]})
    pkf = fl.PurgedKFold(n_splits=3)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([1, 2], [0]), ([0, 2], [1]), ([0, 1], [2])]

def test_PurgedKFold_2_with_purge_without_embargo():
    df = pd.DataFrame({'A': [0, 1, 2, 3]})
    pkf = fl.PurgedKFold(n_splits=2, n_overlaps=1)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([2, 3], [0]), ([0, 1], [3])]

def test_PurgedKFold_3_with_purge_without_embargo():
    df = pd.DataFrame({'A': [0, 1, 2, 3, 4, 5]})
    pkf = fl.PurgedKFold(n_splits=3, n_overlaps=1)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([3, 4, 5], [0, 1]), ([0, 5], [2, 3]), ([0, 1, 2], [4, 5])]

def test_PurgedKFold_2_with_embargo_without_purge():
    df = pd.DataFrame({'A': [0, 1, 2, 3]})
    pkf = fl.PurgedKFold(n_splits=2, n_overlaps=0, pct_embargo=0.25)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([2, 3], [0]), ([0, 1], [2, 3])]

def test_PurgedKFold_2_with_purge_embargo():
    df = pd.DataFrame({'A': [0, 1, 2, 3]})
    pkf = fl.PurgedKFold(n_splits=2, n_overlaps=1, pct_embargo=0.25)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([3], [0]), ([0, 1], [3])]

def test_PurgedKFold_3_with_purge_embargo():
    df = pd.DataFrame({'A': [0, 1, 2, 3, 4, 5, 6]})
    pkf = fl.PurgedKFold(n_splits=3, n_overlaps=1, pct_embargo=0.16)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([4, 5, 6], [0, 1]), ([0, 1, 6], [3]), ([0, 1, 2, 3], [5, 6])]

def test_PurgedKFold_2_special_index():
    df = pd.DataFrame({'A': [0, 1, 2, 3]})
    df.index = [0, 10, 100, 1000]
    pkf = fl.PurgedKFold(n_splits=2)
    idxs = list(map(lambda x: (list(x[0]), list(x[1])), pkf.split(df)))
    assert idxs == [([100, 1000], [0, 10]), ([0, 10], [100, 1000])]
