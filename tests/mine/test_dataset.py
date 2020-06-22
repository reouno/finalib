import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))  # type: ignore
parentdir = os.path.dirname(os.path.split(currentdir)[0])
sys.path.insert(0, parentdir)

import finalib.mine as mi  # pylint: disable=import-error

def test_train_data():
    data = mi.Dataset(([1], []), ([2], []), ([3], []))
    assert data.train_data == ([1], [])

def test_validation_data():
    data = mi.Dataset(([1], []), ([2], []), ([3], []))
    assert data.validation_data == ([2], [])

def test_test_data():
    data = mi.Dataset(([1], []), ([2], []), ([3], []))
    assert data.test_data == ([3], [])
