import numpy as np
import pandas as pd

from abc import ABC
from typing import Union

__all__ = ['Outputs']


class Outputs(ABC):

    def get(self) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        pass
