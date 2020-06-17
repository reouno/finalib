import numpy as np
import pandas as pd

from abc import ABC
from typing import Tuple

from Inputs import Inputs
from Outputs import Outputs

__all__ = ['Data']


class Data(ABC):

    def get(self) -> Tuple[Inputs, Outputs]:
        pass

    def inputs(self) -> Inputs:
        pass

    def outputs(self) -> Outputs:
        pass
