from abc import ABC, abstractmethod
from typing import Any, Dict

from .Inputs import Inputs
from .Outputs import Outputs
from .Score import Score

__all__ = ["Model"]


class Model(ABC):
    """estimator model
    """

    @abstractmethod
    def fit(self, Inputs, Outputs) -> None:
        pass

    @abstractmethod
    def predict(self, Inputs) -> Outputs:
        pass

    @property
    def score(self) -> Score:
        pass