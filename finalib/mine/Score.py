from abc import ABC, abstractmethod


class Score(ABC):
    """score to evaluate predictions
    """

    @abstractmethod
    def compare(self, score: Score) -> int:
        pass
