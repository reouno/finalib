from __future__ import annotations

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from enum import Enum
from sklearn.metrics import accuracy_score
from typing import List, Optional, Union

__all__ = ['Score', 'AccuracyScore', 'Scoring', 'create_score']

Array1DLike = Union[List, np.ndarray, pd.DataFrame, pd.Series]


class Score(ABC):
    """score to evaluate predictions
    """

    def compare(self, score: Score) -> int:
        if (type(self) != type(score)):
            raise ValueError(
                f'Self ({type(self)}) and compared score ({type(score)}) must be the same.')

        return self._compare(score)

    @abstractmethod
    def _compare(self, score: Score) -> int:
        pass

    @abstractmethod
    def calc_score(self, y_true: Array1DLike, y_pred: Array1DLike) -> int:
        pass

    @property
    @abstractmethod
    def value_(self) -> float:
        pass


class AccuracyScore(Score):
    _value_: Optional[int] = None

    def __init__(self):
        pass

    def _compare(self, score: Score) -> int:
        if self.value_ > score.value_:
            return 1
        elif self.value_ == score.value_:
            return 0
        else:
            return -1

    def calc_score(self, y_true: Array1DLike, y_pred: Array1DLike) -> int:
        self._value_ = accuracy_score(y_true, y_pred)
        return self.value_

    @property
    def value_(self):
        if self._value_ is None:
            raise RuntimeError('not calculated yet')

        return self._value_


class Scoring(Enum):
    Accuracy = 0
    Precision = 1
    Recall = 2
    F1 = 3


def create_score(scoring: Scoring):
    if scoring == Scoring.Accuracy:
        return AccuracyScore()
    else:
        raise NotImplementedError(f'score for "{scoring}" is not implemented')
