from abc import ABC, abstractmethod
from itertools import product
from typing import Iterable, Optional

from .estimator import Estimator
from .dataset import Dataset
from .score import Score, Scoring, create_score

__all__ = ['MineUnit', 'Miner', 'ClassifierMiner']


class MineUnit:
    """fit estimator and calculate score to compare performance with other estimators
    """

    def __init__(self, estimator: Estimator, data: Dataset, scoring: Scoring):
        self._estimator: Estimator = estimator
        self._data: Dataset = data
        self._score: Score = create_score(scoring)

    def mine(self) -> None:
        X, y = self._data.train_data
        self._estimator.fit(X, y)
        X_test, y_test = self._data.validation_data
        y_pred = self._estimator.predict(X_test)
        self._score.calc_score(y_test, y_pred)

    @property
    def best_score_(self) -> Score:
        return self._score


class Miner(ABC):

    @abstractmethod
    def mine(self):
        pass

    @property
    @abstractmethod
    def best_result_(self) -> MineUnit:
        pass


class ClassifierMiner(Miner):
    _best_result_: Optional[MineUnit] = None

    def __init__(self,
                 estimators: Iterable[Estimator],
                 datasets: Iterable[Dataset],
                 scoring: Scoring):
        self._estimators: Iterable[Estimator] = estimators
        self._datasets: Iterable[Dataset] = datasets
        self._scoring: Scoring = scoring

    def mine(self):
        for estimator, data in product(self._estimators, self._datasets):
            mineUnit = MineUnit(estimator, data, self._scoring)
            mineUnit.mine()
            # TODO: output summary
            if self._best_result_ is None or \
                    mineUnit.best_score_.compare(self._best_result_.best_score_) > 0:
                self._best_result_ = mineUnit

    @property
    def best_result_(self) -> MineUnit:
        if self._best_result_ is None:
            raise RuntimeError('not mined yet')

        return self._best_result_
