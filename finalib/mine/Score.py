from abc import ABC, abstractmethod


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
