import numpy as np
import pandas as pd

from abc import ABC
from typing import Union

__all__ = ['Inputs']


class Inputs(ABC):

    def get(self) -> Union[np.ndarray, pd.DataFrame]:
        pass
