import numpy as np
import pandas as pd

from typing import List, Tuple, Union

__all__ = ['Dataset']

ArrayLike = Union[List, np.ndarray, pd.DataFrame]
Array1DLike = Union[List, np.ndarray, pd.DataFrame, pd.Series]
DataPair = Tuple[ArrayLike, Array1DLike]


class Dataset:

    def __init__(self, train_data: DataPair, validation_data: DataPair, test_data: DataPair):
        self._train_data: DataPair = train_data
        self._validation_data: DataPair = validation_data
        self._test_data: DataPair = test_data

    @property
    def train_data(self) -> DataPair:
        return self._train_data

    @property
    def validation_data(self) -> DataPair:
        return self._validation_data

    @property
    def test_data(self) -> DataPair:
        return self._test_data
