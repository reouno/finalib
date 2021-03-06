import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection._split import _BaseKFold
from typing import List as List
from typing import Tuple as Tuple
from typing import Union as Union
from debtcollector import moves
warnings.simplefilter('always')

__all__ = ['make_nbars_past', 'make_nbars', 'make_nbars_future', 'split_data', 'PurgedKFold']

def make_nbars_past(df: pd.DataFrame, n_bars: int, cols: List[str] = ['Close'], datetime_col: Union[str, None] = 'Date') -> pd.DataFrame:
    """Make n bars dataframe seeing past n bars.
    The row size of `df` must be greater than or equal to `n_bars`, or raise ValueError.
    """
    if df.shape[0] < n_bars + 1:
        raise ValueError(
            f'row size of the df (={df.shape[0]}) must be greater than or equal to n_bars + 1 (={n_bars + 1})')
    df = df.rename(columns={col: f'{col}{n_bars}' for col in cols})

    for i in reversed(range(n_bars)):
        inc = n_bars - i
        for col in cols:
            df[f'{col}{i}'] = df[f'{col}{n_bars}'][inc:].append(
                pd.Series([np.nan]*inc)).reset_index(drop=True)

    # correct bar date (or datetime)
    if datetime_col is not None:
        df[datetime_col] = df[datetime_col][n_bars:].append(
            pd.Series([np.nan]*n_bars)).reset_index(drop=True)

    df = df.dropna()

    return df


make_nbars = moves.moved_function(make_nbars_past, 'make_nbars', __name__)


def make_nbars_future(df: pd.DataFrame, n_bars: int, cols: List[str] = ['Close'], datetime_col: Union[str, None] = 'Date') -> pd.DataFrame:
    """Make n bars dataframe seeing future n bars.
    The row size of `df` must be greater than or equal to `n_bars`, or raise ValueError.

    Args:
        df (DataFrame): target data frame.
        n_bars (int): number of bars.
        cols (List[str], optional): column names. Defaults to ['Close'].
        datetime_col (Union[str, None], optional): datetime column name. Defaults to 'Date'.

    Raises:
        ValueError: The error is raised when the row size of `df` is smaller than `n_bars`.

    Returns:
        DataFrame: data that can see future n bars
    """
    if df.shape[0] < n_bars + 1:
        raise ValueError(
            f'row size of the df (={df.shape[0]}) must be greater than or equal to n_bars + 1 (={n_bars + 1})')
    df = df.rename(columns={col: f'{col}0' for col in cols})

    for i in range(1, n_bars+1):
        for col in cols:
            df[f'{col}{i}'] = df[f'{col}0'][i:].append(
                pd.Series([np.nan]*i)).reset_index(drop=True)

    df = df.dropna()

    return df


def split_data(df: pd.DataFrame, ratio: float, purging: bool = True, n_bars: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """split data into two

    Args:
        df (DataFrame): Time-seriese data that is to splitted.
        ratio (float): Ratio of the first sub data of the splitted data.
        purging (bool, optional): Purge samples with overlapping periods. Defaults to True.
        n_bars (int, optional): Period of one sample. Defaults to 10.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Pair of the splitted data.
    """
    split_idx = int(df.shape[0] * ratio)
    df1 = df[:split_idx]
    df2 = df[split_idx:]
    if purging:
        purge_idx = round((n_bars-1) * ratio)
        df1 = df1[:-purge_idx]
        df2 = df2[(n_bars - 1 - purge_idx):]

    return df1, df2


class PurgedKFold(_BaseKFold):
    """K-Fold with purging and embargo for finantial sequence data
    """

    def __init__(self, n_splits=5, n_overlaps=0, pct_embargo=0.):
        """constructor

        Args:
            n_splits (int, optional): K of the K-fold. Defaults to 5.
            n_overlaps (int, optional): Temporal overlap between adjacent samples in which samples are purged. Defaults to 0.
            pct_embargo ([type], optional): Percent of embargo. Defaults to 0..
        """
        super(PurgedKFold, self).__init__(
            n_splits, shuffle=False, random_state=None)
        self.n_overlaps = n_overlaps
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        train_ratio = (self.n_splits - 1) / self.n_splits
        indices = np.arange(X.shape[0])
        n_embargo = int(X.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1]+1)
                       for i in np.array_split(indices, self.n_splits)]
        for i, j in test_starts:
            if i != 0 and self.n_overlaps > 0:
                train_f_purge = round(self.n_overlaps * train_ratio)
                train_f_idx1 = i - train_f_purge
                test_idx0 = i + (self.n_overlaps - train_f_purge)
            else:
                train_f_idx1 = test_idx0 = i

            purge_range = self.n_overlaps + n_embargo
            if j != X.shape[0] and purge_range > 0:
                train_l_purge = round(purge_range * train_ratio)
                test_idx1 = j - (purge_range - train_l_purge)
                train_l_idx0 = j + train_l_purge
            else:
                test_idx1 = train_l_idx0 = j

            train_f_indices = pd.Series(X.iloc[0:train_f_idx1].index)
            test_indices = pd.Series(X.iloc[test_idx0:test_idx1].index)
            train_l_indices = pd.Series(X.iloc[train_l_idx0:].index)
            train_indices = pd.concat([train_f_indices, train_l_indices])
            yield train_indices, test_indices
