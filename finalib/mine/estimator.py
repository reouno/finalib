import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from .score import Score

__all__ = ['Estimator', 'SKLEstimator']

Input = Union[np.ndarray, pd.DataFrame]
Output = Union[np.ndarray, pd.DataFrame, pd.Series]


class Estimator(ABC):
    """estimator model
    """

    @abstractmethod
    def fit(self, X: Input, y: Output) -> None:
        pass

    @abstractmethod
    def predict(self, X: Input) -> Output:
        pass


class SKLEstimator(Estimator):
    """estimator for sklearn
    """

    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, X: Input, y: Output):
        return self._estimator.fit(X, y)

    def predict(self, X: Input) -> Output:
        return self._estimator.predict(X)
